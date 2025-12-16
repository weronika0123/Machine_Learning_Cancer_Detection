# Phase 3 - Hyperparameter Search & Model Optimization 🚀

## Przegląd

Phase 3 projektu implementuje kompleksowy system wyszukiwania optymalnych hiperparametrów, analizy wyników eksperymentów oraz demonstracji najlepszego modelu. Ten etap spełnia wszystkie wymagania Phase 3 projektu DNN for Data Analysis.

---

## 📁 Struktura Projektu

```
Machine_Learning_Cancer_Detection/
├── src/
│   ├── main.py                      # [Istniejący] Pipeline treningowy
│   ├── models.py                    # [Istniejący] Definicje modeli
│   ├── hyperparameter_search.py     # [NOWY] Random search hiperparametrów
│   ├── analyze_experiments.py       # [NOWY] Analiza i wizualizacje
│   └── demo.py                      # [NOWY] Demo aplikacja
├── run_phase3.py                    # [NOWY] Master skrypt orkiestrujący
├── experiments/                     # [AUTO-GENEROWANY]
│   ├── experiments_log.csv          # Wyniki wszystkich eksperymentów
│   ├── best_run_info.json           # Konfiguracja najlepszego modelu
│   ├── best_run_report.md           # Raport najlepszego modelu
│   ├── experiments_summary.md       # Podsumowanie wszystkich eksperymentów
│   ├── top10_runs.csv               # TOP 10 najlepszych modeli
│   └── plots/                       # Wizualizacje (8 plików PNG)
├── output/                          # Logi treningowe i modele
└── runs/                            # TensorBoard logs
```

---

## 🚀 Szybki Start

### Opcja 1: Uruchomienie wszystkiego (Rekomendowane)

```bash
python run_phase3.py
```

Ten skrypt automatycznie:
1. Przeprowadzi 25 eksperymentów z różnymi hiperparametrami (~4-6 godzin)
2. Przeanalizuje wyniki i wygeneruje wizualizacje
3. Stworzy raporty i zidentyfikuje najlepszy model

### Opcja 2: Krok po kroku

```bash
# Krok 1: Hyperparameter Search
python src/hyperparameter_search.py

# Krok 2: Analiza wyników
python src/analyze_experiments.py

# Krok 3: Demo najlepszego modelu
python src/demo.py --test_set
```

---

## 📊 Komponenty

### 1. Hyperparameter Search (`hyperparameter_search.py`)

**Przestrzeń przeszukiwania:**
- **Hidden Layers:** 8 architektur ([128,64], [256,128,64], [512,256], etc.)
- **Learning Rate:** 6 wartości (0.0001 - 0.005)
- **Dropout Rate:** 5 wartości (0.1 - 0.5)
- **Batch Size:** 4 wartości (16, 32, 64, 128)
- **Activation:** 4 funkcje (relu, elu, selu, tanh)
- **Epochs:** 3 wartości (50, 75, 100)

**Output:**
- `experiments/experiments_log.csv` - Wszystkie wyniki
- `experiments/best_run_info.json` - Konfiguracja najlepszego modelu

### 2. Experiment Analysis (`analyze_experiments.py`)

Generuje:
- **8 wykresów** pokazujących wpływ każdego hiperparametru na metryki
- **TOP 10 table** najlepszych eksperymentów
- **Correlation heatmap** między hiperparametrami a metrykami
- **Performance timeline** pokazujący postęp
- **Markdown report** z wnioskami i rekomendacjami

**Wykresy:**
1. `hparam_learning_rate.png` - Learning rate vs Performance
2. `hparam_dropout_rate.png` - Dropout rate vs Performance
3. `hparam_batch_size.png` - Batch size vs Performance
4. `hparam_activation.png` - Activation function comparison
5. `hparam_architecture.png` - Architecture comparison
6. `hparam_epochs.png` - Epochs impact
7. `correlation_heatmap.png` - Correlation matrix
8. `performance_timeline.png` - Performance across experiments

### 3. Demo Application (`demo.py`)

Interactive demo najlepszego wytrenowanego modelu.

**Użycie:**

```bash
# Predykcja dla konkretnej próbki
python src/demo.py --sample 10

# Ewaluacja na całym zbiorze testowym
python src/demo.py --test_set

# Tryb interaktywny
python src/demo.py --interactive

# Domyślnie: pokazuje pierwszych 5 próbek
python src/demo.py
```

**Funkcjonalności:**
- Wczytuje najlepszy model z eksperymentów
- Pokazuje predykcje z prawdopodobieństwami
- Wyświetla TOP 10 najważniejszych cech dla każdej próbki
- Generuje confusion matrix
- Oblicza wszystkie metryki (accuracy, precision, recall, F1)

---

## 📈 Metryki i Ewaluacja

Wszystkie eksperymenty są oceniane według:
- **AUC ROC** (primary metric)
- **Accuracy**
- **F1 Score**
- **Precision**
- **Recall**
- **AUC PR**

---

## 🎯 Wyniki Phase 3

Po zakończeniu `run_phase3.py` otrzymasz:

### 1. Experiments Log (`experiments_log.csv`)
Tabela ze wszystkimi eksperymentami zawierająca:
- ID eksperymentu i timestamp
- Wszystkie hiperparametry
- Wszystkie metryki
- Status (success/failed)

### 2. Analysis Reports
- `experiments_summary.md` - Comprehensive analysis z wnioskami
- `best_run_report.md` - Szczegółowy raport najlepszego modelu
- `top10_runs.csv` - TOP 10 najlepszych konfiguracji

### 3. Visualizations (`experiments/plots/`)
8 wykresów wysokiej rozdzielczości (300 DPI) gotowych do prezentacji

### 4. Best Model
- Zapisany w `output/.../[timestamp]/model_best.keras`
- Metadata w `experiments/best_run_info.json`
- Training logs w odpowiednim folderze

### 5. TensorBoard Logs
```bash
tensorboard --logdir runs
```
Uruchom TensorBoard aby zobaczyć:
- Training/validation curves dla wszystkich eksperymentów
- HParams dashboard z porównaniem hiperparametrów
- Model graphs

---

## 💡 Wskazówki

### Optymalizacja czasu

Jeśli chcesz szybsze eksperymenty:
1. Edytuj `src/hyperparameter_search.py`
2. Zmień `n_experiments = 25` na mniejszą liczbę (np. 10)
3. Zmniejsz liczby epoch w `SEARCH_SPACE['epochs']`

### Modyfikacja przestrzeni przeszukiwania

W pliku `src/hyperparameter_search.py` edytuj słownik `SEARCH_SPACE`:

```python
SEARCH_SPACE = {
    'hidden_layers': [[128, 64], [256, 128]],  # Dodaj własne architektury
    'learning_rate': [0.001, 0.0001],          # Dostosuj zakres
    # ...
}
```

### Analiza pojedynczych eksperymentów

```python
import pandas as pd

# Wczytaj wyniki
df = pd.read_csv('experiments/experiments_log.csv')

# Znajdź najlepszy learning rate
best_lr = df.groupby('learning_rate')['auc_roc'].mean().idxmax()
print(f"Best learning rate: {best_lr}")
```

---

## 🎓 Dla Prezentacji Phase 3

Przygotuj:

1. **Slajdy z wynikami:**
   - Użyj wykresów z `experiments/plots/`
   - Pokaż TOP 10 table
   - Wyróżnij najlepszą konfigurację

2. **Live Demo:**
   ```bash
   python src/demo.py --test_set
   ```

3. **TensorBoard:**
   ```bash
   tensorboard --logdir runs
   ```
   Pokaż porównanie eksperymentów w czasie rzeczywistym

4. **Wnioski z `experiments_summary.md`:**
   - Który hiperparametr ma największy wpływ?
   - Jakie są rekomendacje?
   - Jak długo trwały eksperymenty?

---

## 🐛 Troubleshooting

### Problem: "experiments_log.csv not found"
**Rozwiązanie:** Najpierw uruchom `python src/hyperparameter_search.py`

### Problem: "Model file not found"
**Rozwiązanie:** Upewnij się, że przynajmniej jeden eksperyment zakończył się sukcesem i stworzył plik `model_best.keras`

### Problem: Brak pamięci GPU
**Rozwiązanie:** 
1. Zmniejsz batch_size w search space
2. Użyj mniejszych architektur (np. [64, 32])
3. Zmniejsz liczbę epoch

### Problem: Eksperymenty trwają za długo
**Rozwiązanie:**
1. Zmniejsz `n_experiments` w `hyperparameter_search.py`
2. Użyj mniejszych wartości epochs (np. 30 zamiast 100)
3. Pomiń niektóre funkcje aktywacji

---

## 📦 Wymagane Biblioteki

Wszystkie zależności są już zainstalowane dla Phase 1 & 2:
- TensorFlow >= 2.x
- scikit-learn
- pandas
- matplotlib
- seaborn
- numpy

---

## ✅ Checklist Phase 3

- [x] Hyperparameter search zaimplementowany (random search)
- [x] 25+ eksperymentów różnych konfiguracji
- [x] Automatyczne logowanie do TensorBoard
- [x] Wizualizacja wpływu hiperparametrów na metryki
- [x] Identyfikacja najlepszego modelu
- [x] Demo aplikacja CLI
- [x] Comprehensive reports (MD + CSV)
- [x] TOP 10 najlepszych runów
- [x] Model artifacts (best model saved)
- [x] Correlation analysis
- [x] Ready for presentation

---

## 📞 Kontakt

W przypadku pytań dotyczących Phase 3, sprawdź:
1. `experiments/experiments_summary.md` - Automatyczne wnioski
2. `experiments/best_run_report.md` - Info o najlepszym modelu
3. TensorBoard logs - `tensorboard --logdir runs`

---

## 🎉 Gratulacje!

Ukończyłeś implementację Phase 3! Teraz masz:
- ✅ Kompleksowy system wyszukiwania hiperparametrów
- ✅ Automatyczną analizę i wizualizację wyników
- ✅ Demo aplikację gotową do prezentacji
- ✅ Szczegółowe raporty i logi
- ✅ Najlepszy wytrenowany model

**Good luck with your presentation! 🚀**

---

*Dokumentacja wygenerowana dla Phase 3 projektu Deep Neural Networks for Data Analysis*
