# Emocje – System rekomendacji oparty na emocjach w tekście

##  Opis projektu

`Emocje` to projekt w języku C# wykorzystujący **ML.NET** do analizy tekstów i przewidywania emocji wyrażanych w recenzjach lub opiniach.  
Projekt trenuje model klasyfikacji wieloklasowej, który rozpoznaje sześć emocji:  

- sadness  
- anger  
- love  
- surprise  
- fear  
- joy  
Do treningu modelu wykorzystano zbiór danych: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
Model może przewidzieć top 3 emocje dla dowolnego tekstu oraz ocenić skuteczność klasyfikacji na danych walidacyjnych i testowych.

---

##  Funkcjonalności

- Ładowanie danych treningowych, walidacyjnych i testowych z plików `.txt`.  
- Trenowanie modelu klasyfikacji emocji przy użyciu **LightGBM** i reprezentacji n-gramów.  
- Prognozowanie top 3 emocji dla dowolnego tekstu.  
- Ewaluacja modelu: dokładność mikro/makro, log-loss, macierz pomyłek, skuteczność dla każdej klasy i ważona dokładność.  
- Obsługa wag dla rzadziej występujących emocji w danych treningowych.  

---

##  Struktura projektu

```
emocje/
│
├─ Program.cs                # Punkt wejścia aplikacji
├─ DataLoader.cs             # Ładowanie danych z plików
├─ EmotionModel.cs           # Predykcja emocji
├─ EmotionModelTrainer.cs    # Trenowanie modelu ML
├─ ModelEvaluator.cs         # Ewaluacja modelu
│
├─ Models/
│   ├─ TextData.cs           # Klasa danych tekstowych
│   ├─ WeightedData.cs       # Tekst z wagą do treningu
│   └─ EmotionPrediction.cs  # Wynik predykcji emocji
│
├─ train.txt                 # Dane treningowe
├─ val.txt                   # Dane walidacyjne
├─ test.txt                  # Dane testowe
└─ emotion_model.zip         # Zapisany model ML (po treningu)
```

---

##  Jak uruchomić projekt

1. **Sklonuj repozytorium:**
```bash
git clone https://github.com/rogutmichal/emocje.git
```

2. **Otwórz projekt w Visual Studio** wybierając plik `emocje.sln`.

3. **Zainstaluj wymagane pakiety NuGet**:  
   - Microsoft.ML  
   - Microsoft.ML.LightGbm  

4. **Uruchom projekt**  
   - Jeśli model nie istnieje, zostanie wytrenowany automatycznie na danych z `train.txt`.  
   - Po wytrenowaniu model jest zapisany w `emotion_model.zip` i może być ponownie użyty.

5. **Prognozowanie emocji dla przykładowego tekstu** odbywa się w `Program.cs` (przykład: `"I love the animation style!"`).

---

##  Ewaluacja modelu

- Raporty zawierają dokładność mikro/makro, log-loss i macierz pomyłek.  
- Wyświetlana jest skuteczność dla każdej emocji.  
- Specjalna analiza pokazuje np. przypadki, gdy emocja `joy` została błędnie sklasyfikowana jako `love`.  
- Wyliczana jest również **ważona dokładność**, uwzględniająca liczbę próbek dla każdej emocji.

---

##  Jak działa model

1. **Przetwarzanie tekstu:** normalizacja, tokenizacja, usuwanie stop-words, tworzenie n-gramów (1-3).  
2. **Konwersja tokenów na wartości numeryczne (feature vectors).**  
3. **Trenowanie LightGBM** na danych z uwzględnieniem wag dla niedoreprezentowanych emocji.  
4. **Predykcja:** model zwraca prawdopodobieństwa dla wszystkich 6 emocji i wybiera top 3.  

---



Projekt open-source. Możesz dowolnie korzystać z kodu, modyfikować go i rozbudowywać.
