from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QLineEdit, QComboBox, QInputDialog, QLabel, QPushButton, QWidget
)
from PyQt5.QtGui import QColor, QFont, QIcon
from PyQt5.QtCore import QTimer
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from plyer import notification
import winsound

class CryptoDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("داشبورد رمزارزها - نوبیتکس")
        self.setGeometry(100, 100, 1300, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.search_label = QLabel("جستجو:")
        self.search_input = QLineEdit(self)
        self.search_input.textChanged.connect(self.filter_table)

        self.market_selector = QComboBox(self)
        self.market_selector.addItem("تمام بازارها")
        self.market_selector.addItem("تتر/رمزارز")
        self.market_selector.addItem("ریال/رمزارز")
        self.market_selector.currentTextChanged.connect(self.filter_table)

        self.layout.addWidget(self.search_label)
        self.layout.addWidget(self.search_input)
        self.layout.addWidget(self.market_selector)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)
        self.table.setColumnCount(16)  # افزایش تعداد ستون‌ها به 16 تا ستون جدید را در نظر بگیریم
        self.table.setHorizontalHeaderLabels([  
            "نماد بازار", "قیمت لحظه‌ای", "بیشترین", "کمترین",
            "خرید", "فروش", "تغییر روزانه (%)",
            "تغییر لحظه‌ای", "پرش‌های >۲٪ (۱ دقیقه)", "تاریخچه", "پیش‌بینی قیمت", 
            "پیش‌بینی درصد", "هشدار قیمت", "راستی آزمایی (درست/غلط)", "روند بازار"
        ])

        self.price_history = defaultdict(list)
        self.price_alerts = {}
        self.prediction_results = defaultdict(lambda: {'correct': 0, 'incorrect': 0})

        self.update_table()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_table)
        self.timer.start(10000)  # آپدیت هر 10 ثانیه

    def update_table(self):
        try:
            response = requests.get("https://api.nobitex.ir/market/stats")
            response.raise_for_status()
            data = response.json()
            stats = data["stats"]
        except Exception as e:
            print(f"خطا در دریافت داده‌ها: {e}")
            return

        sorted_stats = sorted(stats.items(), key=lambda x: float(x[1].get("dayChange", 0)), reverse=True)
        top_5_stats = sorted_stats[:5]

        self.table.setRowCount(len(top_5_stats))

        for row, (symbol, market_data) in enumerate(top_5_stats):
            try:
                latest = float(market_data["latest"])
                high = float(market_data["dayHigh"])
                low = float(market_data["dayLow"])
                buy = float(market_data["bestBuy"])
                sell = float(market_data["bestSell"])
                change = float(market_data["dayChange"])
            except KeyError:
                continue

            # اصلاح: قیمت لحظه‌ای باید درست محاسبه شود
            last_30_seconds_avg = self.get_30_second_avg(symbol, latest)
            moment_change = ((latest - last_30_seconds_avg) / last_30_seconds_avg) * 100 if last_30_seconds_avg else 0

            self.store_price(symbol, latest)

            self.table.setItem(row, 0, QTableWidgetItem(symbol))
            self.table.setItem(row, 1, self.color_cell(f"{latest:,.0f}", change))
            self.table.setItem(row, 2, QTableWidgetItem(f"{high:,.0f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{low:,.0f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{buy:,.0f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{sell:,.0f}"))
            self.table.setItem(row, 6, self.color_cell(f"{change:.2f}%", change))

            moment_text = f"{moment_change:+.2f}% ({latest - last_30_seconds_avg:+,.0f})"
            self.table.setItem(row, 7, self.color_cell(moment_text, moment_change))

            jumps = self.count_price_jumps(symbol)
            jump_text = f"{jumps} بار پرش" if jumps > 0 else "—"
            self.table.setItem(row, 8, self.color_cell(jump_text, jumps))

            history_text = "\n".join([f"{t.strftime('%H:%M:%S')} - {p}" for t, p in self.price_history[symbol]])
            self.table.setItem(row, 9, QTableWidgetItem(history_text))

            if symbol in self.price_alerts:
                alert_price = self.price_alerts[symbol]
                if latest >= alert_price:
                    notification.notify(
                        title=f"هشدار قیمت {symbol}",
                        message=f"قیمت {symbol} به حد هشدار {alert_price:,.0f} رسیده است.",
                        timeout=5
                    )
                    winsound.Beep(1000, 700)

            predicted_price = self.predict_price(symbol)
            if predicted_price:
                self.table.setItem(row, 10, QTableWidgetItem(f"{int(predicted_price):,}"))
                predicted_percentage = ((predicted_price - latest) / latest) * 100
                self.table.setItem(row, 11, self.color_cell(f"{predicted_percentage:+.2f}%", predicted_percentage))

            alert_button = QPushButton("تنظیم هشدار قیمت")
            alert_button.clicked.connect(lambda _, sym=symbol: self.set_price_alert(sym))
            self.table.setCellWidget(row, 12, alert_button)

            # Column for Verification - Showing correct and incorrect predictions
            correct = self.prediction_results[symbol]['correct']
            incorrect = self.prediction_results[symbol]['incorrect']
            self.table.setItem(row, 13, QTableWidgetItem(f"درست: {correct}, غلط: {incorrect}"))

            # Column for Market Trend
            trend = self.get_market_trend(symbol, latest)
            trend_item = QTableWidgetItem(trend["text"])
            trend_item.setIcon(trend["icon"])
            self.table.setItem(row, 14, trend_item)

        # Set column width for 'راستی آزمایی' to ensure it fits the content properly
        self.table.setColumnWidth(13, 200)  # Adjust this value as needed

    def get_30_second_avg(self, symbol, latest_price):
        now = datetime.now()
        self.price_history[symbol].append((now, latest_price))
        thirty_seconds_ago = now - timedelta(seconds=30)
        self.price_history[symbol] = [
            (t, p) for t, p in self.price_history[symbol] if t >= thirty_seconds_ago
        ]
        prices = [p for t, p in self.price_history[symbol]]
        return np.mean(prices) if prices else None

    def get_market_trend(self, symbol, latest_price):
        last_30_seconds_avg = self.get_30_second_avg(symbol, latest_price)
        if not last_30_seconds_avg:
            return {"text": "—", "icon": QIcon()}
        
        change = ((latest_price - last_30_seconds_avg) / last_30_seconds_avg) * 100
        if change > 2:
            return {"text": "روند صعودی", "icon": QIcon("icons/up_arrow.png")}
        elif change < -2:
            return {"text": "روند نزولی", "icon": QIcon("icons/down_arrow.png")}
        else:
            return {"text": "ثبات", "icon": QIcon("icons/flat_arrow.png")}

    def store_price(self, symbol, price):
        now = datetime.now()
        self.price_history[symbol].append((now, price))
        one_minute_ago = now - timedelta(minutes=1)
        self.price_history[symbol] = [
            (t, p) for t, p in self.price_history[symbol] if t >= one_minute_ago
        ]

    def count_price_jumps(self, symbol):
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return 0
        jumps = 0
        for i in range(1, len(history)):
            prev = history[i - 1][1]
            curr = history[i][1]
            if prev == 0:
                continue
            if abs((curr - prev) / prev) * 100 >= 2:
                jumps += 1
        return jumps

    def color_cell(self, text, change):
        item = QTableWidgetItem(text)
        if isinstance(change, (int, float)):
            # Color green for positive values, red for negative
            if change > 0:
                item.setForeground(QColor("green"))
            elif change < 0:
                item.setForeground(QColor("red"))
            else:
                item.setForeground(QColor("white"))
        item.setBackground(QColor("#333333"))
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        item.setTextAlignment(4)
        return item

    def filter_table(self):
        search_text = self.search_input.text().lower()
        market_filter = self.market_selector.currentText()
        for row in range(self.table.rowCount()):
            symbol_item = self.table.item(row, 0)
            symbol = symbol_item.text().lower() if symbol_item else ""
            self.table.setRowHidden(row, not (
                search_text in symbol and (market_filter == "تمام بازارها" or market_filter in symbol)
            ))

    def set_price_alert(self, symbol):
        alert_price, ok = QInputDialog.getDouble(self, "هشدار قیمت", f"قیمت هشدار برای {symbol} را وارد کنید:")
        if ok:
            self.price_alerts[symbol] = alert_price
            print(f"هشدار برای {symbol} تنظیم شد: {alert_price}")

    def predict_price(self, symbol):
        history = self.price_history.get(symbol, [])
        if len(history) < 5:
            return None

        times = np.array([t.timestamp() for t, _ in history]).reshape(-1, 1)
        prices = np.array([p for _, p in history]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        times_scaled = scaler.fit_transform(times)

        model = self.load_model(symbol)
        if not model:
            return None
        predicted_time = np.array([[times_scaled[-1][0] + 60]])
        predicted_time_scaled = scaler.inverse_transform(predicted_time)
        predicted_price = model.predict(predicted_time_scaled)
        return predicted_price[0][0]

    def load_model(self, symbol):
        try:
            model = tf.keras.models.load_model(f"models/{symbol}_model.h5")
            return model
        except:
            return None


if __name__ == "__main__":
    app = QApplication([])
    window = CryptoDashboard()
    window.show()
    app.exec_()
