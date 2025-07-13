
# Kalp Krizi Prediction API (FastAPI + Railway)

## Açıklama
Bu API, kullanıcının sağlık verilerine göre kalp krizi riski tahmini yapar ve Gemini API desteği ile Türkçe açıklama/öneriler üretebilir.  
Railway üzerinde FastAPI framework kullanılarak deploy edilmiştir ve yapılan tahminleri PostgreSQL veritabanına kaydeder.

## Base URL
https://kalpkrizibackendmodel-production.up.railway.app

## Kullanılabilir Endpoint'ler

### GET `/`
Healthcheck endpoint.

Örnek Response:
```json
{ 
  "message": "Unified backend with /predict + /ask-ai running on Railway!" 
}
```

### POST `/predict`
Sağlık verilerini JSON formatında göndererek tahmin sonucu alınır.  
Opsiyonel olarak `?explain=true` query parametresi kullanarak Gemini API'den açıklama ve öneriler de istenebilir.

Headers:
Content-Type: application/json

Request örneği:
```json
{
  "age": 60,
  "sex": 1,
  "cp": 0,
  "trtbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalachh": 150,
  "exng": 0,
  "oldpeak": 1.2,
  "slp": 1,
  "caa": 0,
  "thall": 2
}
```

Response örneği (`?explain=true` ile):
```json
{
  "prediction": 1,
  "prediction_probability": 0.6493,
  "outcome_message": "Yüksek kalp krizi riski tespit edildi (Güven: 64.93%)",
  "explanation": "Kullanıcının sağlık verilerine göre kalp krizi riski yüksektir. Sigara kullanımını bırakması, sağlıklı beslenmesi ve düzenli egzersiz yapması önerilir."
}
```

### POST `/ask-ai`
Serbest metin sorularını Gemini API üzerinden cevaplar.

Headers:
Content-Type: application/json

Request örneği:
```json
{
  "question": "Kalp krizi risk faktörleri nelerdir?"
}
```

Response örneği:
```json
{
  "answer": "Kalp krizi risk faktörleri arasında hipertansiyon, yüksek kolesterol, sigara kullanımı, diyabet ve obezite bulunur."
}
```

## Frontend Takımı İçin Notlar
- API CORS desteği açık (`allow_origins=["*"]`).
- JSON body formatı yukarıdaki örneklere uygun olmalıdır.
- Tüm response'lar JSON formatındadır.
- Content-Type: application/json header'ı mutlaka gönderilmelidir.

## Environment Variables
- DATABASE_URL: Railway PostgreSQL bağlantısı için gerekli.
- GOOGLE_API_KEY: Gemini API için gerekli Google API Key.

## Kullanılan Teknolojiler
- FastAPI
- SQLAlchemy + PostgreSQL
- Railway deploy ortamı
- Python pickle (model ve scaler yüklemek için)
- Google Gemini API entegrasyonu
