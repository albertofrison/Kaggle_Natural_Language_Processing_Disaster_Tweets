gc() # Garbage collector, to free RAM

#### 1. Metadati del Progetto ####
# SPIEGAZIONE LOGICA: 
# Dichiarazione delle informazioni principali per identificare il lavoro,
# garantire la tracciabilità delle analisi e la riproducibilità del codice.
DATA_PROGETTO <- Sys.Date() # Data corrente
AUTHOR_NAME <- "Alberto FRISON" # Nome dell'autore
GITHUB_ADDRESS <- "https://github.com/albertofrison/" # Indirizzo GitHub
REPO_NAME <- "Kaggle_Natural_Language_Processing_Disaster_Tweets" # Nome repository

#### 2. Importazione Librerie ####
# SPIEGAZIONE LOGICA: 
# Caricamento dei pacchetti R necessari per la gestione dei dati, l'estrazione 
# degli n-grammi, il calcolo TF-IDF, la modellazione regolarizzata LASSO e la visualizzazione.
library(tidyverse) # Manipolazione ed elaborazione dati
library(tidytext) # Tokenizzazione, n-grammi e pesi TF-IDF
library(paletteer) # Tavolozze cromatiche per ggplot2
library(scales) # Formattazione numerica ed assi
library(glmnet) # Modellazione di regressione logistica penalizzata LASSO

#### 3. Pulizia Ambiente e Memoria ####
# SPIEGAZIONE LOGICA: 
# Reset dello spazio di lavoro per eliminare variabili residue, preservando i dataset 
# ufficiali caricati in memoria e liberando la RAM.
rm(list = setdiff(ls(), c("DATA_PROGETTO", "AUTHOR_NAME", "GITHUB_ADDRESS", "REPO_NAME", "train_df", "test_df"))) # Preserva dati
gc() # Libera memoria RAM

#### 4. Definizione Costanti ####
# SPIEGAZIONE LOGICA: 
# Configurazione centralizzata dei parametri del progetto. La costante SOGLIA_DECISIONALE
# viene ottimizzata per incrementare la Recall e bilanciare l'F1-Score.
DEFAULT_PALETTE <- "nationalparkcolors::Acadia" # Palette grafica predefinita
PATH_DATA <- "data/" # Cartella origine dati
PATH_SUBMISSION_DIR <- "submissions/" # Cartella sottomissioni
NUM_PAROLE_VOCABOLARIO <- 1800 # Dimensione vocabolario ibrido
SOGLIA_DECISIONALE <- 0.32 # Soglia ricalibrata per aumentare la Recall
MIN_OCCORRENZE_KW <- 3 # Occorrenze minime per il filtraggio
CUSTOM_STOP_WORDS <- c("http", "https", "amp", "gt", "rt", "youtube", "video", "body", "lol", "im", "vt", "full") # Stop words

#### 5. Importazione Dati e Feature Engineering Strutturale ####
# SPIEGAZIONE LOGICA: 
# Caricamento dei dati ed estrazione delle metriche fisiche/strutturali del testo 
# (lunghezza caratteri, conteggio parole, numeri, link URL, hashtag e menzioni).
if (!exists("train_df")) train_df <- read_csv(paste0(PATH_DATA, "train.csv")) # Caricamento condizionale train
if (!exists("test_df")) test_df <- read_csv(paste0(PATH_DATA, "test.csv")) # Caricamento condizionale test

estrai_feature_strutturali <- function(df) {
  df %>%
    rename(tweet_id = id, target_disaster = any_of("target")) %>% # Standardizzazione colonne
    mutate(
      lunghezza_testo = nchar(text), # Lunghezza in caratteri
      num_parole = str_count(text, "\\w+"), # Conteggio parole
      num_cifre = str_count(text, "\\d"), # Conteggio cifre numeriche
      ha_url = if_else(str_detect(text, "http|https"), 1, 0), # Indicatore presenze URL
      ha_hashtag = if_else(str_detect(text, "#"), 1, 0), # Indicatore presenze hashtag
      ha_menzione = if_else(str_detect(text, "@"), 1, 0) # Indicatore presenze menzioni
    )
}

train_fe <- estrai_feature_strutturali(train_df)
test_fe <- estrai_feature_strutturali(test_df)

#### 6. Estrazione Ibrida Token (Unigrammi + Bigrammi) ####
# SPIEGAZIONE LOGICA: 
# Generazione simultanea di singole parole informative (unigrammi) e coppie di parole 
# adiacenti (bigrammi) depurate dalle stop words per non perdere né contesto né concetti chiave.
data("stop_words") # Caricamento stop words standard

dizionario_stop_words <- bind_rows(
  stop_words,
  tibble(word = CUSTOM_STOP_WORDS, lexicon = "custom")
) # Unione stop words

# Estrazione unigrammi
unigrammi <- train_fe %>%
  select(tweet_id, text, target_disaster) %>%
  unnest_tokens(token, text, token = "words") %>%
  anti_join(dizionario_stop_words, by = c("token" = "word")) %>%
  filter(str_detect(token, "^[a-z]+$"))

# Estrazione bigrammi
bigrammi <- train_fe %>%
  select(tweet_id, text, target_disaster) %>%
  unnest_tokens(token, text, token = "ngrams", n = 2) %>%
  separate(token, c("p1", "p2"), sep = " ") %>%
  filter(!p1 %in% dizionario_stop_words$word, !p2 %in% dizionario_stop_words$word) %>%
  filter(str_detect(p1, "^[a-z]+$"), str_detect(p2, "^[a-z]+$")) %>%
  unite(token, p1, p2, sep = " ")

# Unione ibrida dei token
tokens_ibridi <- bind_rows(unigrammi, bigrammi)

#### 7. Selezione Vocabolario Discriminativo (Log Odds Ratio) ####
# SPIEGAZIONE LOGICA: 
# Calcolo della forza associativa dei token ibridi mediante Log Odds Ratio 
# per selezionare le entità a maggior potere predittivo basandosi su NUM_PAROLE_VOCABOLARIO.
vocabolario_selezionato <- tokens_ibridi %>%
  count(target_disaster, token) %>%
  pivot_wider(names_from = target_disaster, values_from = n, values_fill = 0) %>%
  rename(n_disastro = `1`, n_non_disastro = `0`) %>%
  filter((n_disastro + n_non_disastro) >= MIN_OCCORRENZE_KW) %>% # Filtro frequenza minima
  mutate(
    log_odds = log((n_disastro + 1) / (sum(n_disastro) + 1)) -
      log((n_non_disastro + 1) / (sum(n_non_disastro) + 1)) # Calcolo Log Odds
  ) %>%
  slice_max(abs(log_odds), n = NUM_PAROLE_VOCABOLARIO) # Selezione basata su NUM_PAROLE_VOCABOLARIO

#### 8. Costruzione Matrice DTM/TF-IDF ed Integrazione Feature ####
# SPIEGAZIONE LOGICA: 
# Trasformazione dei dati testuali in pesi rilevanti tramite TF-IDF ed unione con 
# le caratteristiche strutturali estratte in precedenza.
matrice_tfidf <- tokens_ibridi %>%
  filter(token %in% vocabolario_selezionato$token) %>%
  count(tweet_id, token) %>%
  bind_tf_idf(token, tweet_id, n) %>% # Calcolo pesi TF-IDF
  pivot_wider(id_cols = tweet_id, names_from = token, values_from = tf_idf, values_fill = 0)

train_completo <- train_fe %>%
  select(tweet_id, target_disaster, lunghezza_testo, num_parole, num_cifre, ha_url, ha_hashtag, ha_menzione) %>%
  left_join(matrice_tfidf, by = "tweet_id") %>%
  mutate(across(-c(tweet_id, target_disaster), ~ replace_na(.x, 0))) # Imputazione zeri

#### 9. Addestramento Modello LASSO Regolarizzato ####
# SPIEGAZIONE LOGICA: 
# Stima del modello di regressione logistica con penalizzazione L1 (LASSO) 
# per selezionare automaticamente le feature ed eliminare quelle ridondanti.
x_train <- as.matrix(train_completo %>% select(-tweet_id, -target_disaster))
y_train <- train_completo$target_disaster

modello_lasso_ibrido <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial", # Classificazione binaria
  alpha = 1 # Penalizzazione LASSO L1
)

#### 10. Calcolo Metriche KPI con Soglia Decisionale Ottimizzata ####
# SPIEGAZIONE LOGICA: 
# Calcolo delle probabilità stimate e applicazione della nuova SOGLIA_DECISIONALE 
# per recuperare la Recall ed elevare il valore complessivo dell'F1-Score.
prob_train <- predict(modello_lasso_ibrido, newx = x_train, s = "lambda.1se", type = "response")
pred_train <- if_else(as.numeric(prob_train) >= SOGLIA_DECISIONALE, 1, 0) # Applicazione SOGLIA_DECISIONALE

tp <- sum(y_train == 1 & pred_train == 1)
fp <- sum(y_train == 0 & pred_train == 1)
fn <- sum(y_train == 1 & pred_train == 0)

precision <- tp / (tp + fp)
recall <- tp / (tp + fn)
f1_score <- 2 * (precision * recall) / (precision + recall)

tabella_kpi <- tibble(
  metrica = factor(c("Precision", "Recall", "F1-Score"), levels = c("Precision", "Recall", "F1-Score")),
  valore = c(precision, recall, f1_score)
)

#### 11. Visualizzazione Grafica Prestazioni KPI Ribilanciate ####
# SPIEGAZIONE LOGICA: 
# Generazione del grafico comparativo finale delle metriche di performance tramite ggplot2 
# utilizzando la palette DEFAULT_PALETTE.
grafico_kpi_ribilanciato <- tabella_kpi %>%
  ggplot(aes(x = metrica, y = valore, fill = metrica)) +
  geom_col(show.legend = FALSE, width = 0.4) + # Tracciamento barre verticali
  geom_text(aes(label = percent(valore, accuracy = 0.1)), vjust = -0.5, size = 4.5, fontface = "bold") + # Etichette percentuali
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) + # Formattazione asse y
  scale_fill_paletteer_d(DEFAULT_PALETTE) + # Applicazione DEFAULT_PALETTE
  labs(
    title = "Prestazioni del Modello LASSO Ibrido con TF-IDF e Soglia Ottimizzata",
    subtitle = "Effetto dell'integrazione di unigrammi, bigrammi e ricalibrazione della soglia decisionale",
    x = "Indicatori di Performance (KPI)",
    y = "Punteggio / Percentuale (%)",
    caption = paste0("Fonte: Dataset Kaggle Disaster Tweets | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 11, face = "bold"))

print(grafico_kpi_ribilanciato)




################################################################################
gc() # Garbage collector, to free RAM

#### 1. Metadati del Progetto ####
# SPIEGAZIONE LOGICA: 
# Dichiarazione delle informazioni principali per identificazione e riproducibilita.
# Preserva l'attribuibilita e la tracciabilita del codice.
DATA_PROGETTO <- Sys.Date() # Data corrente
AUTHOR_NAME <- "Alberto FRISON" # Nome dell'autore
GITHUB_ADDRESS <- "https://github.com/albertofrison/" # Indirizzo GitHub
REPO_NAME <- "Kaggle_Natural_Language_Processing_Disaster_Tweets" # Nome repository

#### 2. Importazione Librerie ####
# SPIEGAZIONE LOGICA: 
# Caricamento dei pacchetti R necessari per la gestione delle strutture dati,
# l'estrazione delle metriche, la modellazione e la grafica vettoriale.
library(tidyverse) # Manipolazione ed elaborazione dati
library(tidytext) # Tokenizzazione, n-grammi e pesi TF-IDF
library(paletteer) # Tavolozze cromatiche
library(scales) # Formattazione percentuale ed assi
library(glmnet) # Modello di regressione penalizzata LASSO

#### 3. Pulizia Ambiente e Memoria ####
# SPIEGAZIONE LOGICA: 
# Ripristino dello spazio di lavoro e liberazione della memoria RAM,
# preservando i dati originali ed il modello precedentemente addestrato.
rm(list = setdiff(ls(), c("DATA_PROGETTO", "AUTHOR_NAME", "GITHUB_ADDRESS", "REPO_NAME", 
                          "train_df", "test_df", "modello_lasso_ibrido", "train_completo", 
                          "vocabolario_selezionato", "dizionario_stop_words"))) # Preserva dati
gc() # Libera la memoria RAM

#### 4. Definizione Costanti ####
# SPIEGAZIONE LOGICA: 
# Configurazione accentrata dei parametri di ricerca e delle costanti grafiche.
# L'intervallo ed il passo del ciclo for sono definiti da variabili capitalizzate.
DEFAULT_PALETTE <- "nationalparkcolors::Acadia" # Palette grafica predefinita
PATH_DATA <- "data/" # Cartella origine dati
PATH_SUBMISSION_DIR <- "submissions/" # Cartella sottomissioni
PASSO_SOGLIA <- 0.01 # Incremento della soglia nel ciclo for
SOGLIA_MIN <- 0.05 # Soglia minima della griglia
SOGLIA_MAX <- 0.95 # Soglia massima della griglia

#### 5. Estrazione Probabilita ed Inizializzazione Griglia ####
# SPIEGAZIONE LOGICA: 
# Calcoliamo le probabilita continue sul train set tramite il modello LASSO ibrido
# e generiamo il vettore delle soglie candidate che verranno valutate nel ciclo for.
x_train <- as.matrix(train_completo %>% select(-tweet_id, -target_disaster)) # Matrice feature
y_train <- train_completo$target_disaster # Vettore target reale

prob_train <- predict(modello_lasso_ibrido, newx = x_train, s = "lambda.1se", type = "response") # Probabilita stimate

griglia_soglie <- seq(from = SOGLIA_MIN, to = SOGLIA_MAX, by = PASSO_SOGLIA) # Generazione sequenza soglie

#### 6. Ciclo For per la Ricerca della Soglia che Massimizza F1 ####
# SPIEGAZIONE LOGICA: 
# Iteriamo su ogni soglia della griglia. Ad ogni passaggio del ciclo for binarizziamo 
# le probabilita, calcoliamo la matrice di confusione e salviamo Precision, Recall ed F1-Score.
risultati_soglie <- tibble() # Dataframe vuoto per memorizzazione

for (soglia in griglia_soglie) {
  pred_temp <- if_else(as.numeric(prob_train) >= soglia, 1, 0) # Classificazione temporanea
  
  tp_temp <- sum(y_train == 1 & pred_temp == 1) # Veri positivi
  fp_temp <- sum(y_train == 0 & pred_temp == 1) # Falsi positivi
  fn_temp <- sum(y_train == 1 & pred_temp == 0) # Falsi negativi
  
  precision_temp <- if_else((tp_temp + fp_temp) > 0, tp_temp / (tp_temp + fp_temp), 0) # Precision temporanea
  recall_temp <- if_else((tp_temp + fn_temp) > 0, tp_temp / (tp_temp + fn_temp), 0) # Recall temporanea
  f1_temp <- if_else((precision_temp + recall_temp) > 0, 
                     2 * (precision_temp * recall_temp) / (precision_temp + recall_temp), 0) # F1 temporaneo
  
  risultati_soglie <- bind_rows(
    risultati_soglie,
    tibble(
      soglia = soglia,
      precision = precision_temp,
      recall = recall_temp,
      f1_score = f1_temp
    )
  )
}

#### 7. Estrazione e Selezione della Soglia Ottimale ####
# SPIEGAZIONE LOGICA: 
# Isoliamo la riga del dataframe in cui l'F1-Score raggiunge il punto di massimo 
# ed assegniamo la soglia corrispondente alla variabile SOGLIA_OTTIMALE.
riga_ottimale <- risultati_soglie %>% 
  slice_max(f1_score, n = 1, with_ties = FALSE) # Estrazione riga con F1 massimo

SOGLIA_OTTIMALE <- riga_ottimale$soglia # Assegnazione della soglia ottimale
F1_MASSIMO <- riga_ottimale$f1_score # Assegnazione del punteggio F1 massimo

#### 8. Visualizzazione Grafica: Curva di Ottimizzazione F1-Score vs Soglia ####
# SPIEGAZIONE LOGICA: 
# Generiamo un grafico a linee con ggplot2 per mostrare l'andamento di Precision, Recall 
# ed F1-Score al variare della soglia, evidenziando il punto di massimo con una linea verticale.
grafico_curva_ottimizzazione <- risultati_soglie %>%
  pivot_longer(cols = c(precision, recall, f1_score), names_to = "metrica", values_to = "valore") %>%
  mutate(metrica = factor(metrica, levels = c("precision", "recall", "f1_score"),
                          labels = c("Precision", "Recall", "F1-Score"))) %>%
  ggplot(aes(x = soglia, y = valore, color = metrica)) +
  geom_line(linewidth = 1.2) + # Tracciamento linee metriche
  geom_vline(xintercept = SOGLIA_OTTIMALE, linetype = "dashed", color = "black", linewidth = 0.8) + # Intercetta soglia ottimale
  annotate("text", x = SOGLIA_OTTIMALE + 0.02, y = 0.25, 
           label = paste0("Soglia Ottimale = ", percent(SOGLIA_OTTIMALE, accuracy = 0.1)), 
           hjust = 0, fontface = "bold", color = "black") + # Etichetta valore ottimale
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.05)) + # Formattazione percentuale asse y
  scale_x_continuous(labels = percent_format(accuracy = 1), breaks = seq(0, 1, 0.1)) + # Formattazione percentuale asse x
  scale_color_paletteer_d(DEFAULT_PALETTE) + # Applicazione DEFAULT_PALETTE
  labs(
    title = "Ottimizzazione della Soglia Decisionale tramite Grid Search Ciclo For",
    subtitle = "Compromesso tra Precision, Recall ed F1-Score al variare della soglia nel modello LASSO",
    x = "Soglia Decisionale Valutata (%)",
    y = "Punteggio / Percentuale (%)",
    color = "Indicatore KPI",
    caption = paste0("Fonte: Grid Search Ciclo For | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() +
  theme(legend.position = "top", axis.text = element_text(size = 10, face = "bold"))

print(grafico_curva_ottimizzazione)

#### 9. Calcolo Metriche KPI Finali con SOGLIA_OTTIMALE ####
# SPIEGAZIONE LOGICA: 
# Calcoliamo gli indicatori di prestazione definitivi applicando SOGLIA_OTTIMALE 
# e strutturiamo la tabella riassuntiva per la rappresentazione grafica a barre.
pred_train_ottimale <- if_else(as.numeric(prob_train) >= SOGLIA_OTTIMALE, 1, 0) # Classificazione ottimale

tp_opt <- sum(y_train == 1 & pred_train_ottimale == 1)
fp_opt <- sum(y_train == 0 & pred_train_ottimale == 1)
fn_opt <- sum(y_train == 1 & pred_train_ottimale == 0)

precision_opt <- tp_opt / (tp_opt + fp_opt)
recall_opt <- tp_opt / (tp_opt + fn_opt)
f1_score_opt <- 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)

tabella_kpi_ottimali <- tibble(
  metrica = factor(c("Precision", "Recall", "F1-Score"), levels = c("Precision", "Recall", "F1-Score")),
  valore = c(precision_opt, recall_opt, f1_score_opt)
)

#### 10. Visualizzazione Grafica Metriche KPI Ottimizzate ####
# SPIEGAZIONE LOGICA: 
# Costruiamo il grafico a barre a colori differenziati che sintetizza i valori finali 
# di Precision, Recall ed F1-Score ottenuti in corrispondenza di SOGLIA_OTTIMALE.
grafico_kpi_ottimizzati <- tabella_kpi_ottimali %>%
  ggplot(aes(x = metrica, y = valore, fill = metrica)) +
  geom_col(show.legend = FALSE, width = 0.4) + # Tracciamento barre verticali
  geom_text(aes(label = percent(valore, accuracy = 0.1)), vjust = -0.5, size = 4.5, fontface = "bold") + # Etichette percentuali
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) + # Asse y formattato
  scale_fill_paletteer_d(DEFAULT_PALETTE) + # Applicazione DEFAULT_PALETTE
  labs(
    title = "Metriche di Performance Ottimizzate con SOGLIA_OTTIMALE",
    subtitle = "Valori di Precision, Recall ed F1-Score calcolati in corrispondenza del punto di massimo",
    x = "Indicatori di Performance (KPI)",
    y = "Punteggio / Percentuale (%)",
    caption = paste0("Fonte: Modellazione LASSO Ibrida | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 11, face = "bold"))

print(grafico_kpi_ottimizzati)