gc() # Garbage collector, to free RAM

#### 1. Metadati del Progetto ####
# SPIEGAZIONE LOGICA: 
# Questa sezione definisce le meta-informazioni del progetto. Serve a garantire
# la riproducibilità e a tracciare l'autore e la fonte del codice, elementi 
# fondamentali nelle buone pratiche di Data Science.
DATA_PROGETTO <- Sys.Date() # Data corrente
AUTHOR_NAME <- "Alberto FRISON" # Nome dell'autore
GITHUB_ADDRESS <- "https://github.com/albertofrison/" # Indirizzo GitHub
REPO_NAME <- "Kaggle_Natural_Language_Processing_Disaster_Tweets" # Nome della repository

#### 2. Importazione Librerie ####
# SPIEGAZIONE LOGICA: 
# Qui carichiamo gli "attrezzi del mestiere". Usiamo pacchetti specifici per 
# manipolare dati, elaborare testi (NLP), creare grafici accattivanti e 
# addestrare modelli statistici avanzati sfruttando tutti i core del computer.
library(tidyverse) # Manipolazione dati, trasformazioni ed importazione CSV
library(tidytext) # Tokenizzazione ed elaborazione del testo
library(paletteer) # Gestione tavolozze cromatiche
library(scales) # Formattazione numerica e percentuali negli assi
library(glmnet) # Modelli di regressione logistica penalizzata LASSO
library(doParallel) # Gestione dell'elaborazione parallela e multithreading
library(foreach) # Cicli paralleli per ottimizzazione del tempo di calcolo

#### 3. Pulizia Ambiente e Memoria ####
# SPIEGAZIONE LOGICA: 
# Svuotiamo l'ambiente di lavoro di R da variabili residue di sessioni precedenti. 
# Questo previene errori logici causati da dati vecchi in memoria, mantenendo 
# solo le costanti di configurazione definite al passo 1.
rm(list = setdiff(ls(), c("DATA_PROGETTO", "AUTHOR_NAME", "GITHUB_ADDRESS", "REPO_NAME"))) # Preserva metadati
gc() # Libera la memoria RAM

#### 4. Definizione Costanti ####
# SPIEGAZIONE LOGICA: 
# Definiamo tutte le costanti (parametri che non cambiano) in un unico posto. 
# Questo rende il codice "modulare": se in futuro vorrai cambiare la soglia 
# di decisione o il numero di parole, ti basterà modificare questi valori 
# senza dover cercare nel resto del codice.
DEFAULT_PALETTE <- "nationalparkcolors::Acadia" # Palette grafica predefinita
PATH_DATA <- "data/" # Cartella di origine dei file CSV
PATH_SUBMISSION_DIR <- "submissions/" # Cartella di destinazione delle submission
NUM_PAROLE_VOCABOLARIO <- 1500 # Dimensione del vocabolario ottimale
SOGLIA_DECISIONALE <- 0.5 # Soglia di probabilita per la classificazione
MIN_OCCORRENZE_KW <- 10 # Soglia minima occorrenze
TOP_N_KEYWORDS <- 25 # Numero keyword da mostrare nei grafici
TOP_N_PAROLE <- 20 # Numero parole per i grafici Log Odds
NUM_CORES <- max(1, parallel::detectCores() - 1) # Core CPU per multithreading
KAGGLE_LEADERBOARD_SCORE <- 0.79589 # Punteggio F1 di riferimento
CUSTOM_STOP_WORDS <- c("http", "https", "amp", "gt", "rt", "youtube", "video", "body", "lol", "im", "vt", "full") # Stop words

#### 5. Importazione Dataset ####
# SPIEGAZIONE LOGICA: 
# Carichiamo i dati grezzi in memoria. Abbiamo un set di addestramento 
# su cui il modello imparerà, e un set di test su cui faremo le previsioni finali.
train_df <- read_csv(paste0(PATH_DATA, "train.csv")) # Carica dataset addestramento
test_df <- read_csv(paste0(PATH_DATA, "test.csv")) # Carica dataset di test

#### 6. EDA: Analisi del Bilanciamento della Variabile Target ####
# SPIEGAZIONE LOGICA: 
# Controlliamo se abbiamo lo stesso numero di tweet per i veri disastri e per 
# i non disastri. Se le classi fossero molto sbilanciate (es. 99% vs 1%), il 
# modello tenderebbe a prevedere sempre la classe maggioritaria.
distribuzione_target <- train_df %>%
  group_by(target) %>% 
  summarise(conteggio = n()) %>%
  mutate(
    etichetta = if_else(target == 1, "Disastro Reale (1)", "Non Disastro (0)"),
    proporzione = conteggio / sum(conteggio)
  )

grafico_target <- distribuzione_target %>%
  ggplot(aes(x = etichetta, y = conteggio, fill = etichetta)) +
  geom_col(show.legend = FALSE, width = 0.5) +
  geom_text(aes(label = percent(proporzione, accuracy = 0.1)), vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_continuous(labels = comma, limits = c(0, max(distribuzione_target$conteggio) * 1.15)) +
  scale_fill_paletteer_d(DEFAULT_PALETTE) +
  labs(
    title = "Distribuzione della Variabile Target",
    subtitle = "Frequenza assoluta e proporzione delle classi nel dataset di addestramento",
    x = "Categoria del Tweet",
    y = "Frequenza Assoluta (N. Tweet)",
    caption = paste0("Fonte: Kaggle | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal()

print(grafico_target)

#### 7. Feature Engineering Strutturale ####
# SPIEGAZIONE LOGICA: 
# Oltre al significato delle parole, la *struttura* di un tweet può indicare 
# un'emergenza. Chi è nel panico potrebbe scrivere tweet più brevi o inserire 
# link a notizie. Questa funzione calcola queste variabili (feature) aggiuntive.
estrai_feature_strutturali <- function(df) {
  df %>%
    rename(tweet_id = id, target_disaster = any_of("target")) %>% # Standardizza nomi
    mutate(
      lunghezza_testo = nchar(text), # Conta caratteri
      num_parole = str_count(text, "\\w+"), # Conta parole
      num_cifre = str_count(text, "\\d"), # Conta numeri
      ha_url = if_else(str_detect(text, "http|https"), 1, 0), # Flag link
      ha_hashtag = if_else(str_detect(text, "#"), 1, 0), # Flag hashtag
      ha_menzione = if_else(str_detect(text, "@"), 1, 0) # Flag menzioni
    )
}

train_fe <- estrai_feature_strutturali(train_df)
test_fe <- estrai_feature_strutturali(test_df)

#### 8. NLP: Selezione Vocabolario tramite Log Odds Ratio ####
# SPIEGAZIONE LOGICA: 
# Non tutte le parole sono utili. Rimuoviamo prima le parole comuni e inutili 
# (stop words). Poi calcoliamo il Log Odds Ratio: una formula statistica che 
# ci dice se una parola appare proporzionalmente molto di più nei tweet di 
# disastro rispetto agli altri. Selezioniamo solo le parole più discriminanti.
data("stop_words")

dizionario_stop_words <- bind_rows(
  stop_words,
  tibble(word = CUSTOM_STOP_WORDS, lexicon = "custom")
)

vocabolario_log_odds <- train_fe %>%
  select(tweet_id, text, target_disaster) %>%
  unnest_tokens(word, text) %>%
  anti_join(dizionario_stop_words, by = "word") %>%
  filter(str_detect(word, "^[a-z]+$")) %>% # Solo lettere
  count(target_disaster, word) %>%
  pivot_wider(names_from = target_disaster, values_from = n, values_fill = 0) %>%
  rename(n_disastro = `1`, n_non_disastro = `0`) %>%
  filter((n_disastro + n_non_disastro) >= MIN_OCCORRENZE_KW) %>% # Applica filtro MIN_OCCORRENZE_KW
  mutate(
    log_odds = log((n_disastro + 1) / (sum(n_disastro) + 1)) -
      log((n_non_disastro + 1) / (sum(n_non_disastro) + 1))
  ) %>%
  slice_max(abs(log_odds), n = NUM_PAROLE_VOCABOLARIO) # Applica limite NUM_PAROLE_VOCABOLARIO

#### 9. Visualizzazione: Top Parole per Log Odds ####
# SPIEGAZIONE LOGICA: 
# Visualizziamo le parole che il modello ritiene più fortemente associate a 
# un disastro reale o a un non-disastro, per verificare che la statistica 
# abbia un senso logico umano.
parole_grafico <- vocabolario_log_odds %>%
  mutate(classe = if_else(log_odds > 0, "Disastro Reale (1)", "Non Disastro (0)")) %>%
  group_by(classe) %>%
  slice_max(abs(log_odds), n = TOP_N_PAROLE) %>%
  ungroup()

grafico_log_odds <- parole_grafico %>%
  ggplot(aes(x = reorder_within(word, log_odds, classe), y = log_odds, fill = classe)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  coord_flip() +
  scale_x_reordered() +
  scale_fill_paletteer_d(DEFAULT_PALETTE) +
  facet_wrap(~ classe, scales = "free") +
  labs(
    title = "Termini Piu Distintivi (Log Odds Ratio)",
    subtitle = "Misura della forza associativa delle parole alle classi target",
    x = "Parola Estratta",
    y = "Log Odds Ratio (Valore Statistico)",
    caption = paste0("Fonte: Dataset NLP | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold"))

print(grafico_log_odds)

#### 10. Costruzione Matrici DTM (Document-Term Matrix) ####
# SPIEGAZIONE LOGICA: 
# Trasformiamo il testo in numeri affinché l'algoritmo matematico possa leggerlo. 
# Creiamo una grande tabella dove ogni riga è un tweet e ogni colonna è una parola 
# del nostro vocabolario. Il valore sarà 1 se la parola è presente, 0 altrimenti.
dtm_train <- train_fe %>%
  select(tweet_id, text) %>%
  unnest_tokens(word, text) %>%
  filter(word %in% vocabolario_log_odds$word) %>%
  distinct(tweet_id, word) %>%
  mutate(presenza = 1) %>%
  pivot_wider(id_cols = tweet_id, names_from = word, values_from = presenza, values_fill = 0)

# Uniamo le feature strutturali con la DTM testuale
train_completo <- train_fe %>%
  select(tweet_id, target_disaster, lunghezza_testo, num_parole, num_cifre, ha_url, ha_hashtag, ha_menzione) %>%
  left_join(dtm_train, by = "tweet_id") %>%
  mutate(across(-c(tweet_id, target_disaster), ~ replace_na(.x, 0)))

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

# Allineamento rigoroso: assicura che il test set abbia le stesse identiche colonne del train set
vocab_colonne <- setdiff(colnames(train_completo), c("tweet_id", "target_disaster"))
colonne_mancanti_test <- setdiff(vocab_colonne, colnames(test_completo))

if (length(colonne_mancanti_test) > 0) {
  matrice_mancante <- matrix(0, nrow = nrow(test_completo), ncol = length(colonne_mancanti_test),
                             dimnames = list(NULL, colonne_mancanti_test))
  test_completo <- bind_cols(test_completo, as_tibble(matrice_mancante))
}
test_completo <- test_completo %>% select(tweet_id, all_of(vocab_colonne))

#### 11. Addestramento Modello LASSO Parallelo ####
# SPIEGAZIONE LOGICA: 
# Addestriamo una regressione logistica penalizzata (LASSO). Questa tecnica, 
# oltre a fare previsioni, "spegne" (azzera) le variabili e le parole che non 
# sono realmente utili, prevenendo l'overfitting (sovradattamento). 
# Attiviamo il multithreading per calcolare su più core contemporaneamente, risparmiando tempo.
cluster_parallelo <- makeCluster(NUM_CORES) 
registerDoParallel(cluster_parallelo) 

x_train_mat <- as.matrix(train_completo %>% select(-tweet_id, -target_disaster))
y_train_vec <- train_completo$target_disaster

modello_lasso_parallel <- cv.glmnet(
  x = x_train_mat,
  y = y_train_vec,
  family = "binomial", # Per target 0 o 1
  alpha = 1, # L1 regularization (LASSO)
  parallel = TRUE 
)

stopCluster(cluster_parallelo) # Liberiamo le risorse del computer

#### 12. Valutazione KPI su Train Set ####
# SPIEGAZIONE LOGICA: 
# Calcoliamo quanto è bravo il nostro modello. L'F1-Score è una media armonica 
# tra Precision (quanti dei previsti disastri lo erano davvero) e Recall 
# (quanti disastri veri siamo riusciti a individuare). È la metrica ufficiale su Kaggle.
prob_train <- predict(modello_lasso_parallel, newx = x_train_mat, s = "lambda.1se", type = "response")
pred_train <- if_else(as.numeric(prob_train) >= SOGLIA_DECISIONALE, 1, 0)

tp <- sum(y_train_vec == 1 & pred_train == 1)
fp <- sum(y_train_vec == 0 & pred_train == 1)
fn <- sum(y_train_vec == 1 & pred_train == 0)

precision <- tp / (tp + fp)
recall <- tp / (tp + fn)
f1_score <- 2 * (precision * recall) / (precision + recall)

tabella_kpi <- tibble(
  metrica = factor(c("Precision", "Recall", "F1-Score"), levels = c("Precision", "Recall", "F1-Score")),
  valore = c(precision, recall, f1_score)
)

grafico_kpi <- tabella_kpi %>%
  ggplot(aes(x = metrica, y = valore, fill = metrica)) +
  geom_col(show.legend = FALSE, width = 0.5) +
  geom_text(aes(label = percent(valore, accuracy = 0.1)), vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1)) +
  scale_fill_paletteer_d(DEFAULT_PALETTE) +
  labs(
    title = "Metriche di Performance Modello LASSO Integrato",
    subtitle = "Valutazione su dati addestramento basata sulle configurazioni ottimali",
    x = "Indicatore (KPI)",
    y = "Punteggio (%)",
    caption = paste0("Fonte: Addestramento Modello | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal()

print(grafico_kpi)

#### 13. Previsioni sul Test Set ed Esportazione ####
# SPIEGAZIONE LOGICA: 
# Ora che il modello è addestrato, gli diamo in pasto dati nuovi e mai visti 
# prima (Test set) per generare le previsioni finali. Salviamo poi questi 
# risultati in un file CSV formattato secondo le regole di Kaggle, aggiungendo 
# data e ora al nome del file per non sovrascrivere tentativi precedenti.
x_test_mat <- as.matrix(test_completo %>% select(-tweet_id))

prob_test <- predict(modello_lasso_parallel, newx = x_test_mat, s = "lambda.1se", type = "response")

submission_final <- test_completo %>%
  select(id = tweet_id) %>%
  mutate(target = if_else(as.numeric(prob_test) >= SOGLIA_DECISIONALE, 1, 0))

if (!dir.exists(PATH_SUBMISSION_DIR)) {
  dir.create(PATH_SUBMISSION_DIR, recursive = TRUE)
}

TIMESTAMP_CORRENTE <- format(Sys.time(), "%Y%m%d_%H%M%S")
PERCORSO_COMPLETO_SUBMISSION <- paste0(PATH_SUBMISSION_DIR, "submission_fe_lasso_", TIMESTAMP_CORRENTE, ".csv")

write_csv(submission_final, PERCORSO_COMPLETO_SUBMISSION)

#### 14. Visualizzazione Distribuzione Previsioni (Test) ####
# SPIEGAZIONE LOGICA: 
# Ultimo controllo visivo: verifichiamo la distribuzione delle etichette predette 
# sui dati di test. Se la distribuzione differisse drasticamente rispetto a quella 
# calcolata nella sezione EDA iniziale, potrebbe esserci un problema nel modello.
grafico_distribuzione_test <- submission_final %>%
  mutate(classe_predetta = if_else(target == 1, "Disastro (1)", "Non Disastro (0)")) %>%
  count(classe_predetta) %>%
  ggplot(aes(x = classe_predetta, y = n, fill = classe_predetta)) +
  geom_col(show.legend = FALSE, width = 0.5) +
  geom_text(aes(label = comma(n)), vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_y_continuous(labels = comma, limits = c(0, max(nrow(submission_final)) * 0.75)) +
  scale_fill_paletteer_d(DEFAULT_PALETTE) +
  labs(
    title = "Previsioni Generate sul Dataset di Test",
    subtitle = paste0("Conteggio classi modello LASSO - Salvato in ", PERCORSO_COMPLETO_SUBMISSION),
    x = "Classe Predetta",
    y = "N. Tweet Predetti",
    caption = paste0("Fonte: Submission Kaggle | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal()

print(grafico_distribuzione_test)