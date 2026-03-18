# 1. TEMEL İMAJ SEÇİMİ
FROM python:3.9-slim

# Neden slim? Normal Python imajı 1GB yer kaplar. 'slim' versiyonu 
# sadece en gerekli dosyaları içerir (yaklaşık 150MB). Sunucuda yer tasarrufu sağlar.

# 2. ÇALIŞMA DİZİNİ
WORKDIR /app

# Konteynerin içinde '/app' diye bir klasör açar ve "bundan sonraki 
# tüm işleri bu klasörün içinde yap" der.

# 3. SİSTEM BAĞIMLILIKLARI
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# NEDEN?: Bizim 'implicit' kütüphanemiz C++ ile yazılmıştır. 
# 'pip install' yaparken bu kütüphaneyi derlemek (compile) için 
# standart bir Linux sisteminde bulunmayan 'build-essential' araçlarına ihtiyacımız var.

# 4. KÜTÜPHANE YÜKLEME
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NEDEN ÖNCE COPY requirements?: Docker "katman" (layer) mantığıyla çalışır. 
# Kodun değişse bile kütüphaneler değişmediyse Docker kütüphaneleri 
# önbellekten (cache) çeker, saniyeler içinde build eder.

# 5. KODLARI KOPYALAMA
COPY . .

# Bilgisayarındaki tüm proje dosyalarını (api, src, models) konteynerin içine atar.

# 6. ORTAM DEĞİŞKENLERİ
ENV PYTHONPATH=/app
ENV OPENBLAS_NUM_THREADS=1

# PYTHONPATH: 'ModuleNotFoundError: No module named src' hatasını Docker içinde 
# kalıcı olarak çözer. 
# OPENBLAS: ALS modelinin performans uyarısını susturur.

# 7. PORT AÇMA
EXPOSE 8000


# 8. ÇALIŞTIRMA KOMUTU

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

