# Levenshtein Mesafesi Hesaplama (Python)

Bu proje, iki kelime arasındaki **Levenshtein mesafesini** hesaplayan basit bir Python uygulamasıdır. Kullanıcıdan alınan iki kelimeye göre, ekleme, silme ve değiştirme işlemlerini temel alarak kelimeler arasındaki farkı sayısal olarak verir. Ayrıca benzerlik oranını da hesaplar.

## Özellikler

- İki kelime arasındaki farkı (Levenshtein distance) hesaplar
- Benzerlik oranını yüzde olarak verir
- NumPy ile matris kullanımı içerir
- Türkçe terminal arayüzü ile çalışır

## Levenshtein Mesafesi Nedir?

Levenshtein mesafesi, bir kelimeyi başka bir kelimeye dönüştürmek için gereken minimum işlem sayısıdır. İşlemler:
- Harf ekleme
- Harf silme
- Harf değiştirme

## 🛠Kullanılan Teknolojiler

- Python 3
- NumPy

## Kurulum

Terminal veya VS Code üzerinden çalıştırabilirsiniz:

```bash
git clone https://github.com/jssavenger/levenshtein-Mesafesi.git
cd levenshtein-Mesafesi
pip install numpy
python levenshtein.py

```
## Örnek Çıktı
1. kelimeyi girin
kitap
2. kelimeyi girin
kütüphane

'kitap     ' ve 'kütüphane' arasindaki Levenshtein Mesafesi:
6.0
Benzerlik orani:
0.333


## Geliştirici
Emin Ocak
