from flask import Flask, render_template
import requests

app = Flask(__name__)

# این تابع برای دریافت قیمت‌ها و جزئیات از CoinGecko است
def get_crypto_prices():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"  # URL برای دریافت قیمت‌ها
    try:[nobitex_code.txt](https://github.com/user-attachments/files/19549985/nobitex_code.txt)

        response = requests.get(url)
        response.raise_for_status()  # بررسی وضعیت پاسخ
        data = response.json()
        return data  # برگشت قیمت‌ها به صورت یک لیست
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# مسیر برای صفحه اصلی که لیست قیمت‌های رمزارزها را نشان می‌دهد
@app.route('/')
def index():
    crypto_prices = get_crypto_prices()  # دریافت قیمت‌ها
    if not crypto_prices:
        return "Error fetching prices", 500
    return render_template('index.html', prices=crypto_prices)

if __name__ == "__main__":
    app.run(debug=True)

