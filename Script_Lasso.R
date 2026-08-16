gc() # Garbage collector, to free RAM

#### 1. Metadati del Progetto ####
# Dichiarazione delle informazioni principali per identificazione e riproducibilita
DATA_PROGETTO <- Sys.Date() # Data corrente
AUTHOR_NAME <- "Alberto FRISON" # Nome dell'autore
GITHUB_ADDRESS <- "https://github.com/albertofrison/" # Indirizzo GitHub
REPO_NAME <- "Kaggle_Natural_Language_Processing_Disaster_Tweets" # Nome della repository

#### 2. Importazione Librerie ####
# Caricamento pacchetti per elaborazione dati, NLP, modellazione e calcolo parallelo
library(tidyverse) # Manipolazione dati e grafica
library(tidytext) # Tokenizzazione ed elaborazione testo
library(paletteer) # Gestione tavolozze colori
library(scales) # Formattazione numerica e percentuali
library(glmnet) # Regressione logistica penalizzata LASSO
library(doParallel) # Elaborazione parallela e multithreading
library(foreach) # Gestione cicli paralleli

#### 3. Pulizia Ambiente e Memoria ####
# Mantenimento dei dati grezzi originali e pulizia della RAM
rm(list = setdiff(ls(), c("DATA_PROGETTO", "AUTHOR_NAME", "GITHUB_ADDRESS", "REPO_NAME", "train_df", "test_df", "sub_df"))) # Preserva dataframe originali
gc() # Libera RAM

#### 4. Definizione Costanti ####
# Configurazione dei parametri di modellazione, risorse hardware e stile grafico
DEFAULT_PALETTE <- "nationalparkcolors::Acadia" # Palette colori predefinita
NUM_PAROLE_VOCABOLARIO <- 1500 # Dimensione del vocabolario di parole chiavi
SOGLIA_DECISIONALE <- 0.5 # Soglia di probabilita per la classificazione binaria
NUM_CORES <- max(1, parallel::detectCores() - 1) # Conteggio core CPU per multithreading
PATH_SUBMISSION_DIR <- "submissions/" # Cartella dedicata alle submission

#### 5. Configurazione del Cluster Parallelo ####
# Inizializzazione ed attivazione del backend multithread
cluster_parallelo <- makeCluster(NUM_CORES) # Creazione del cluster con NUM_CORES
registerDoParallel(cluster_parallelo) # Registrazione backend parallelo per glmnet

#### 6. Feature Engineering e Costruzione DTM Completa ####
# Estrazione variabili strutturali dal testo grezzo per train e test
estrai_feature_strutturali <- function(df) {
  df %>%
    rename(
      tweet_id = id,
      target_disaster = any_of("target") # Gestisce presenza opzionale colonna target
    ) %>%
    mutate(
      lunghezza_testo = nchar(text), # Conteggio caratteri totali
      num_parole = str_count(text, "\\w+"), # Conteggio parole totali
      num_cifre = str_count(text, "\\d"), # Conteggio cifre numeriche
      ha_url = if_else(str_detect(text, "http|https"), 1, 0), # Indicatore presenze link
      ha_hashtag = if_else(str_detect(text, "#"), 1, 0), # Indicatore presenza hashtag
      ha_menzione = if_else(str_detect(text, "@"), 1, 0) # Indicatore presenza menzioni
    )
}

train_fe <- estrai_feature_strutturali(train_df)
test_fe <- estrai_feature_strutturali(test_df)

# Selezione vocabolario principale dalle sole parole di addestramento
parole_top <- train_fe %>%
  select(tweet_id, text) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_detect(word, "^[a-z]+$")) %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = NUM_PAROLE_VOCABOLARIO) # Selezione basata su NUM_PAROLE_VOCABOLARIO

# Creazione matrice DTM per il train set
dtm_train <- train_fe %>%
  select(tweet_id, text) %>%
  unnest_tokens(word, text) %>%
  filter(word %in% parole_top$word) %>%
  distinct(tweet_id, word) %>%
  mutate(presenza = 1) %>%
  pivot_wider(id_cols = tweet_id, names_from = word, values_from = presenza, values_fill = 0)

train_completo <- train_fe %>%
  select(tweet_id, target_disaster, lunghezza_testo, num_parole, num_cifre, ha_url, ha_hashtag, ha_menzione) %>%
  left_join(dtm_train, by = "tweet_id") %>%
  mutate(across(-c(tweet_id, target_disaster), ~ replace_na(.x, 0)))

# Creazione matrice DTM per il test set
dtm_test <- test_fe %>%
  select(tweet_id, text) %>%
  unnest_tokens(word, text) %>%
  filter(word %in% parole_top$word) %>%
  distinct(tweet_id, word) %>%
  mutate(presenza = 1) %>%
  pivot_wider(id_cols = tweet_id, names_from = word, values_from = presenza, values_fill = 0)

test_completo <- test_fe %>%
  select(tweet_id, lunghezza_testo, num_parole, num_cifre, ha_url, ha_hashtag, ha_menzione) %>%
  left_join(dtm_test, by = "tweet_id") %>%
  mutate(across(-tweet_id, ~ replace_na(.x, 0)))

# Allineamento rigoroso delle colonne tra train e test set
vocab_colonne <- setdiff(colnames(train_completo), c("tweet_id", "target_disaster"))
colonne_mancanti_test <- setdiff(vocab_colonne, colnames(test_completo))

if (length(colonne_mancanti_test) > 0) {
  matrice_mancante <- matrix(0, nrow = nrow(test_completo), ncol = length(colonne_mancanti_test),
                             dimnames = list(NULL, colonne_mancanti_test))
  test_completo <- bind_cols(test_completo, as_tibble(matrice_mancante))
}

test_completo <- test_completo %>% select(tweet_id, all_of(vocab_colonne))

#### 7. Stima Modello LASSO con Multithreading ####
# Preparazione delle matrici per l'algoritmo glmnet
x_train_mat <- as.matrix(train_completo %>% select(-tweet_id, -target_disaster))
y_train_vec <- train_completo$target_disaster
x_test_mat <- as.matrix(test_completo %>% select(-tweet_id))

# Misurazione del tempo di esecuzione dell'addestramento parallelo
tempo_inizio <- Sys.time()

modello_lasso_parallel <- cv.glmnet(
  x = x_train_mat,
  y = y_train_vec,
  family = "binomial", # Famiglia binomiale per la classificazione
  alpha = 1, # Penalizzazione L1
  parallel = TRUE # Abilita l'esecuzione su multithread tramite doParallel
)

tempo_fine <- Sys.time()
tempo_calcolo <- as.numeric(difftime(tempo_fine, tempo_inizio, units = "secs"))

# Chiusura del cluster per rilasciare i thread di calcolo
stopCluster(cluster_parallelo)

#### 8. Previsione su Test Set ed Esportazione Versionata ####
# Inferenza del modello sul dataset di test non osservato
prob_test <- predict(
  modello_lasso_parallel,
  newx = x_test_mat,
  s = "lambda.1se", # Selezione della penalizzazione ottimale
  type = "response"
)

submission_fe <- test_completo %>%
  select(id = tweet_id) %>%
  mutate(target = if_else(as.numeric(prob_test) >= SOGLIA_DECISIONALE, 1, 0)) # Applicazione SOGLIA_DECISIONALE

# Salvataggio del file CSV con timestamping e percorso dedicato
if (!dir.exists(PATH_SUBMISSION_DIR)) {
  dir.create(PATH_SUBMISSION_DIR, recursive = TRUE)
}

TIMESTAMP_CORRENTE <- format(Sys.time(), "%Y%m%d_%H%M%S")
NOME_FILE_SUBMISSION <- paste0("submission_disaster_fe_parallel_", TIMESTAMP_CORRENTE, ".csv")
PERCORSO_COMPLETO_SUBMISSION <- paste0(PATH_SUBMISSION_DIR, NOME_FILE_SUBMISSION)

write_csv(submission_fe, PERCORSO_COMPLETO_SUBMISSION)

#### 9. Visualizzazione Grafica dei Tempi di Calcolo Parallelo ####
# Rappresentazione del tempo di esecuzione ottenuto con la configurazione multithread
dati_prestazioni <- tibble(
  metrica = "Tempo Calcolo CV Parallela",
  valore = tempo_calcolo
)

grafico_prestazioni <- dati_prestazioni %>%
  ggplot(aes(x = metrica, y = valore, fill = metrica)) +
  geom_col(show.legend = FALSE, width = 0.3) + # Tracciamento barra
  geom_text(aes(label = paste0(round(valore, 2), " s")), vjust = -0.5, size = 5, fontface = "bold") + # Etichetta tempo
  scale_y_continuous(limits = c(0, max(tempo_calcolo * 1.3, 1))) + # Scaling asse y
  scale_fill_paletteer_d(DEFAULT_PALETTE) + # Applicazione DEFAULT_PALETTE
  labs(
    title = "Prestazioni di Calcolo del Modello LASSO con Multithreading",
    subtitle = "Validazione crociata eseguita in parallelo sfruttando i core definiti da NUM_CORES",
    x = "Operazione Modellistica",
    y = "Tempo di Esecuzione (Secondi)",
    caption = paste0("Fonte: Benchmark Locale | Autore: ", AUTHOR_NAME, " | ", GITHUB_ADDRESS, REPO_NAME)
  ) +
  theme_minimal() + # Tema grafico essenziale
  theme(
    axis.text.x = element_text(size = 11, face = "bold") # Testo asse x in grassetto
  )

# Output del grafico
print(grafico_prestazioni)