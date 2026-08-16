gc() # Garbage collector, to free RAM

#### 1. Metadati del Progetto ####
# SPIEGAZIONE LOGICA:
# Questa sezione definisce le informazioni identificative principali. 
# Mantenere queste informazioni in testa allo script garantisce la tracciabilità 
# del codice e la riproducibilità del flusso di lavoro.
DATA_PROGETTO <- Sys.Date() # Data corrente
AUTHOR_NAME <- "Alberto FRISON" # Nome dell'autore
GITHUB_ADDRESS <- "https://github.com/albertofrison/" # Indirizzo GitHub
REPO_NAME <- "Kaggle_Natural_Language_Processing_Disaster_Tweets" # Nome repository

#### 2. Importazione Librerie ####
# SPIEGAZIONE LOGICA:
# Carichiamo i pacchetti R per la manipolazione dati, l'analisi del testo (NLP), 
# la costruzione dei grafici e l'addestramento ad alte prestazioni di Random Forest via ranger.
library(tidyverse) # Manipolazione dati ed importazione
library(tidytext) # Tokenizzazione ed elaborazione testo
library(paletteer) # Tavolozze cromatiche
library(scales) # Formattazione numerica degli assi
library(ranger) # Implementazione veloce di Random Forest

#### 3. Pulizia Ambiente e Memoria ####
# SPIEGAZIONE LOGICA:
# Preserviamo i dati grezzi se già presenti in memoria e resettiamo il resto 
# dell'ambiente di lavoro per evitare conflitti con variabili o vettori residui.
rm(list = setdiff(ls(), c("DATA_PROGETTO", "AUTHOR_NAME", "GITHUB_ADDRESS", "REPO_NAME", "train_df", "test_df"))) # Preserva metadati e dati
gc() # Libera memoria RAM

#### 4. Definizione Costanti ####
# SPIEGAZIONE LOGICA:
# Configurazione accentrata dei parametri di progetto. Riduciamo leggermente 
# la dimensione del vocabolario rispetto a LASSO per ottimizzare i tempi di calcolo 
# e l'efficienza degli alberi decisionali.
DEFAULT_PALETTE <- "nationalparkcolors::Acadia" # Palette grafica predefinita
PATH_DATA <- "data/" # Cartella dati
PATH_SUBMISSION_DIR <- "submissions/" # Cartella di destinazione submission
NUM_PAROLE_VOCABOLARIO <- 600 # Dimensione vocabolario per Random Forest
SOGLIA_DECISIONALE <- 0.50 # Soglia di probabilita per classificazione
NUM_ALBERI <- 300 # Numero di alberi decisionali nella foresta
MIN_OCCORRENZE_KW <- 5 # Frequenza minima per parole discriminative
NUM_CORES <- max(1, parallel::detectCores() - 1) # Core CPU per calcolo parallelo
CUSTOM_STOP_WORDS <- c("http", "https", "amp", "gt", "rt", "youtube", "video", "body", "lol", "im", "vt", "full") # Stop words

#### 5. Importazione Dati e Feature Engineering ####
# SPIEGAZIONE LOGICA:
# Se i dataframe non sono presenti in memoria, li carichiamo da disco. 
# Successivamente estraiamo le caratteristiche strutturali (lunghezza, numeri, hashtag, URL) 
# che il Random Forest potrà combinare in modo non lineare con il testo.
if (!exists("train_df")) train_df <- read_csv(paste0(PATH_DATA, "train.csv")) # Caricamento condizionale train
if (!exists("test_df")) test_df <- read_csv(paste0(PATH_DATA, "test.csv")) # Caricamento condizionale test

estrai_feature_strutturali <- function(df) {
  df %>%
    rename(tweet_id = id, target_disaster = any_of("target")) %>% # Standardizzazione colonne
    mutate(
      lunghezza_testo = nchar(text), # Lunghezza in caratteri
      num_parole = str_count(text, "\\w+"), # Conteggio parole
      num_cifre = str_count(text, "\\d"), # Conteggio cifre
      ha_url = if_else(str_detect(text, "http|https"), 1, 0), # Indicatore URL
      ha_hashtag = if_else(str_detect(text, "#"), 1, 0), # Indicatore hashtag
      ha_menzione = if_else(str_detect(text, "@"), 1, 0) # Indicatore menzioni
    )
}

train_fe <- estrai_feature_strutturali(train_df)
test_fe <- estrai_feature_strutturali(test_df)

#### 6. Selezione Vocabolario Discriminativo (Log Odds Ratio) ####
# SPIEGAZIONE LOGICA:
# Depuriamo il testo dalle stop words ed estraiamo i termini con il maggior potere 
# discriminativo calcolando il Log Odds Ratio tra la classe Disastro e Non Disastro.
data("stop_words") # Caricamento stop words standard

dizionario_stop_words <- bind_rows(
  stop_words,
  tibble(word = CUSTOM_STOP_WORDS, lexicon = "custom")
) # Unione stop words

vocabolario_log_odds <- train_fe %>%
  select(tweet_id, text, target_disaster) %>%
  unnest_tokens(word, text) %>% # Tokenizzazione
  anti_join(dizionario_stop_words, by = "word") %>% # Filtraggio stop words
  filter(str_detect(word, "^[a-z]+$")) %>% # Selezione sole stringhe alfabetiche
  count(target_disaster, word) %>%
  pivot_wider(names_from = target_disaster, values_from = n, values_fill = 0) %>%
  rename(n_disastro = `1`, n_non_disastro = `0`) %>%
  filter((n_disastro + n_non_disastro) >= MIN_OCCORRENZE_KW) %>% # Filtro frequenza con MIN_OCCORRENZE_KW
  mutate(
    log_odds = log((n_disastro + 1) / (sum(n_disastro) + 1)) -
      log((n_non_disastro + 1) / (sum(n_non_disastro) + 1)) # Calcolo log odds
  ) %>%
  slice_max(abs(log_odds), n = NUM_PAROLE_VOCABOLARIO) # Selezione limitata da NUM_PAROLE_VOCABOLARIO

#### 7. Costruzione DTM ed Allineamento Matrici ####
# SPIEGAZIONE LOGICA:
# Trasformiamo il testo in una matrice binaria di presenza/assenza (DTM) e la 
# uniamo alle caratteristiche strutturali. Allineiamo le colonne del test set 
# affinché corrispondano perfettamente a quelle usate durante l'addestramento.
dtm_train <- train_fe %>%
  select(tweet_id, text) %>%
  unnest_tokens(word, text) %>%
  filter(word %in% vocabolario_log_odds$word) %>%
  distinct(tweet_id, word) %>%
  mutate(presenza = 1) %>%
  pivot_wider(id_cols = tweet_id, names_from = word, values_from = presenza, values_fill = 0)

train_completo <- train_fe %>%
  select(tweet_id, target_disaster, lunghezza_testo, num_parole, num_cifre, ha_url, ha_hashtag, ha_menzione) %>%
  left_join(dtm_train, by = "tweet_id") %>%
  mutate(across(-c(tweet_id, target_disaster), ~ replace_na(.x, 0))) # Imputazione valori mancanti

dtm_test <- test_fe %>%
  select(tweet_id, text) %>%
  unnest_tokens(word, text) %>%
  filter(word %in% vocabolario_log_odds$word) %>%
  distinct(tweet_id, word) %>%
  mutate(presenza = 1) %>%
  pivot_wider(id_cols = tweet_id, names_from = word, values_from = presenza, values_fill = 0)

test_completo <- test_fe %>%
  select(tweet_id, lunghezza_testo, num_parole, num_cifre, ha_url, ha_hashtag, ha_menzione) %>%
  left_join(dtm_test, by = "tweet_id") %>%
  mutate(across(-tweet_id, ~ replace_na(.x, 0)))

# Allineamento rigoroso della struttura dei dati
vocab_colonne <- setdiff(colnames(train_completo), c("tweet_id", "target_disaster"))
colonne_mancanti_test <- setdiff(vocab_colonne, colnames(test_completo))

if (length(colonne_mancanti_test) > 0) {
  matrice_mancante <- matrix(0, nrow = nrow(test_completo), ncol = length(colonne_mancanti_test),
                             dimnames = list(NULL, colonne_mancanti_test))
  test_completo <- bind_cols(test_completo, as_tibble(matrice_mancante))
}
test_completo <- test_completo %>% select(tweet_id, all_of(vocab_colonne))

#### 8. Addestramento Modello Random Forest con ranger ####
# SPIEGAZIONE LOGICA:
# Addestriamo la foresta casuale impiegando la libreria `ranger`. 
# Abilitiamo la stima delle probabilità e sfruttiamo il calcolo multithread 
# definito dal numero di core della CPU.
train_rf_data <- train_completo %>% 
  mutate(target_disaster = factor(target_disaster, levels = c(0, 1))) # Conversione target in fattore

modello_rf <- ranger(
  formula = target_disaster ~ . - tweet_id,
  data = train_rf_data,
  num.trees = NUM_ALBERI, # Impostazione numero alberi con NUM_ALBERI
  probability = TRUE, # Restituisce probabilita anziche classi rigide
  num.threads = NUM_CORES, # Parallelizzazione con NUM_CORES
  seed = 123 # Riproducibilita
)

#### 9. Calcolo Metriche KPI ed F1-Score ####
# SPIEGAZIONE LOGICA:
# Estraiamo le probabilità previste per la classe 1 (Disastro) sui dati di addestramento 
# e binarizziamo l'esito usando la soglia decisionale per calcolare Precision, Recall ed F1-Score.
prob_train_rf <- predict(modello_rf, data = train_rf_data)$predictions[, "1"]
pred_train_rf <- if_else(prob_train_rf >= SOGLIA_DECISIONALE, 1, 0)
y_reale <- train_completo$target_disaster

tp_rf <- sum(y_reale == 1 & pred_train_rf == 1)
fp_rf <- sum(y_reale == 0 & pred_train_rf == 1)
fn_rf <- sum(y_reale == 1 & pred_train_rf == 0)

precision_rf <- tp_rf / (tp_rf + fp_rf)
recall_rf <- tp_rf / (tp_rf + fn_rf)
f1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

tabella_kpi_rf <- tibble(
  metrica = factor(c("Precision", "Recall", "F1-Score"), levels = c("Precision", "Recall", "F1-Score")),
  valore = c(precision_rf, recall_rf, f1_rf)
)

#### 10. Visualizzazione Grafica Prestazioni KPI ####
# SPIEGAZIONE LOGICA:
# Rappresentiamo graficamente gli indicatori di performance del modello Random Forest.
grafico_kpi_rf <- tabella_kpi_rf %>%
  ggplot(aes(x = metrica, y = valore, fill = metrica)) +
  geom_col(show.legend = FALSE, width = 0.4) +
  geom_text(aes(label = percent(valore, accuracy = 0.1)), vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) +
  scale_fill_paletteer_d(DEFAULT_PALETTE) +
  labs(
    title = "Prestazioni del Modello Random Forest (ranger)",
    subtitle = "Metriche di valutazione stimate con probabilità binarizzate sul dataset di addestramento",
    x = "Indicatori di Performance (KPI)",
    y = "Punteggio / Percentuale (%)",
    caption = paste0("Fonte: Modello Random Forest | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 11, face = "bold"))

print(grafico_kpi_rf)

#### 11. Generazione Previsioni su Test Set ed Esportazione ####
# SPIEGAZIONE LOGICA:
# Applichiamo il Random Forest sui dati di test non visti prima, calcoliamo le probabilità, 
# applichiamo la soglia ed esportiamo il file CSV di sottomissione con marca temporale.
prob_test_rf <- predict(modello_rf, data = test_completo)$predictions[, "1"]

submission_rf <- test_completo %>%
  select(id = tweet_id) %>%
  mutate(target = if_else(prob_test_rf >= SOGLIA_DECISIONALE, 1, 0))

if (!dir.exists(PATH_SUBMISSION_DIR)) {
  dir.create(PATH_SUBMISSION_DIR, recursive = TRUE)
}

TIMESTAMP_CORRENTE <- format(Sys.time(), "%Y%m%d_%H%M%S")
PERCORSO_COMPLETO_SUBMISSION <- paste0(PATH_SUBMISSION_DIR, "submission_random_forest_", TIMESTAMP_CORRENTE, ".csv")

write_csv(submission_rf, PERCORSO_COMPLETO_SUBMISSION)

#### 12. Visualizzazione Distribuzione Previsioni Test Set ####
# SPIEGAZIONE LOGICA:
# Generiamo un grafico a barre per verificare la quantità di disastri e non-disastri 
# stimati sul test set dal modello Random Forest.
distribuzione_rf <- submission_rf %>%
  mutate(classe_predetta = if_else(target == 1, "Disastro Reale (1)", "Non Disastro (0)")) %>%
  count(classe_predetta)

grafico_distribuzione_rf <- distribuzione_rf %>%
  ggplot(aes(x = classe_predetta, y = n, fill = classe_predetta)) +
  geom_col(show.legend = FALSE, width = 0.4) +
  geom_text(aes(label = comma(n)), vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_y_continuous(labels = comma, limits = c(0, max(distribuzione_rf$n) * 1.15)) +
  scale_fill_paletteer_d(DEFAULT_PALETTE) +
  labs(
    title = "Distribuzione delle Previsioni sul Test Set (Random Forest)",
    subtitle = paste0("Conteggio delle classi individuate ed esportate in ", PERCORSO_COMPLETO_SUBMISSION),
    x = "Classe Predetta dal Modello",
    y = "Frequenza Assoluta (N. Tweet)",
    caption = paste0("Fonte: Previsioni Random Forest | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 11, face = "bold"))

print(grafico_distribuzione_rf)