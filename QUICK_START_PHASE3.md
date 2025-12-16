# Quick Start Guide - Phase 3 🚀

## Szybkie Uruchomienie

### ⚡ Opcja 1: Automatyczne (Polecane)

```bash
python run_phase3.py
```

Odpowiedz "yes" gdy zostaniesz zapytany. Skrypt automatycznie:
- Uruchomi 25 eksperymentów (~4-6 godzin)
- Przeanalizuje wyniki
- Wygeneruje raporty i wykresy
- Zidentyfikuje najlepszy model

### 🎯 Opcja 2: Ręczne kroki

```bash
# 1. Uruchom hyperparameter search (zajmie kilka godzin)
python src/hyperparameter_search.py

# 2. Przeanalizuj wyniki
python src/analyze_experiments.py

# 3. Przetestuj najlepszy model
python src/demo.py --test_set
```

---

## 📊 Co otrzymujesz?

Po zakończeniu znajdziesz w folderze `experiments/`:

```
experiments/
├── experiments_log.csv          # ← Wszystkie wyniki
├── experiments_summary.md       # ← Analiza i wnioski
├── best_run_report.md           # ← Raport najlepszego modelu
├── best_run_info.json          
├── top10_runs.csv              
└── plots/                       # ← 8 wykresów gotowych do prezentacji
    ├── hparam_learning_rate.png
    ├── hparam_dropout_rate.png
    ├── hparam_batch_size.png
    ├── hparam_activation.png
    ├── hparam_architecture.png
    ├── hparam_epochs.png
    ├── correlation_heatmap.png
    └── performance_timeline.png
```

---

## 🎮 Demo Aplikacja

```bash
# Zobacz pierwszych 5 próbek
python src/demo.py

# Predykcja konkretnej próbki
python src/demo.py --sample 10

# Ewaluacja na całym zbiorze testowym + confusion matrix
python src/demo.py --test_set

# Tryb interaktywny
python src/demo.py --interactive
```

---

## 📈 TensorBoard

```bash
tensorboard --logdir runs
```

Potem otwórz: http://localhost:6006

Zobaczysz:
- Training curves wszystkich eksperymentów
- HParams comparison
- Model architecture

---

## ⏱️ Szybkie eksperymenty (dla testów)

Jeśli chcesz przetestować system szybko:

1. Otwórz `src/hyperparameter_search.py`
2. Znajdź linię: `n_experiments = 25`
3. Zmień na: `n_experiments = 3`
4. Uruchom: `python run_phase3.py`

To zajmie tylko ~30-60 minut zamiast kilku godzin.

---

## 🎓 Dla Prezentacji

### Krok 1: Przygotuj slajdy
Użyj wykresów z `experiments/plots/` i tabel z `experiments_summary.md`

### Krok 2: Live Demo
```bash
python src/demo.py --test_set
```

### Krok 3: Pokaż TensorBoard
```bash
tensorboard --logdir runs
```

### Krok 4: Omów wnioski
Otwórz i omów `experiments/experiments_summary.md`

---

## 🆘 Problemy?

### "experiments_log.csv not found"
→ Uruchom najpierw: `python src/hyperparameter_search.py`

### "Model file not found"  
→ Poczekaj aż zakończy się przynajmniej jeden eksperyment

### Za wolno?
→ Zmniejsz `n_experiments` w `hyperparameter_search.py`

---

## ✅ Checklist przed prezentacją

- [ ] Uruchomiłem `run_phase3.py` i eksperymenty się zakończyły
- [ ] Mam folder `experiments/` z wynikami
- [ ] Przejrzałem `experiments_summary.md`
- [ ] Sprawdziłem wykresy w `experiments/plots/`
- [ ] Przetestowałem `python src/demo.py --test_set`
- [ ] Uruchomiłem TensorBoard i sprawdziłem logi
- [ ] Przygotowałem slajdy z wynikami
- [ ] Znam najlepszą konfigurację hiperparametrów

---

## 📚 Więcej informacji

Pełna dokumentacja: [PHASE3_README.md](PHASE3_README.md)

---

**Good luck! 🎉**
