# Kalp Krizi Tahmin API'si (FastAPI + Railway)

## Açıklama
Bu API, kullanıcının sağlık bilgilerine göre kalp krizi riski tahmini yapar.
FastAPI ile geliştirilmiş, Railway üzerinde deploy edilmiştir ve yapılan tahminleri PostgreSQL veritabanına kaydeder.

## Base URL
https://kalpkrizibackendmodel-production.up.railway.app

## Kullanılabilir Endpoint'ler

### GET /
Healthcheck endpoint:
Yanıt: { "message": "Heart attack prediction API is running on Railway!" }

### POST /predict
Sağlık verilerini JSON formatında göndererek tahmin sonucu alınır.

Örnek Request Body:
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

Örnek Response:
{
  "prediction": 1
}

## Frontend Takımı İçin Notlar
- API CORS desteği açık (allow_origins=["*"]).
- JSON body yukarıdaki örneğe uygun formatta olmalıdır.
- Content-Type: application/json header'ı gönderilmelidir.

## Environment Variables (Ortam Değişkenleri)
- DATABASE_URL: PostgreSQL bağlantı adresi (Railway veritabanı bağlantısı için gerekli).

## Kullanılan Teknolojiler
- FastAPI
- SQLAlchemy + PostgreSQL
- Railway deploy ortamı
- Python pickle (model ve scaler yüklemek için)
