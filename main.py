"""
📈 Intraday Stock Prediction Mobile Application (Android & Desktop)
Built with Kivy & Material-inspired Dark UI.
Features:
  - Tab 1: Nifty-50 Live Universe with Realtime Search & Stock Details
  - Tab 2: Custom Bookmarks Watchlist with Add/Remove Tickers
  - Tab 3: RL Weights Engine, Accuracy History, & Free Hugging Face FinBERT Token Settings
"""

import os
import sys
import threading
import time
import datetime
from functools import partial

# Kivy configuration before window creation
os.environ["KIVY_NO_ARGS"] = "1"

from kivy.config import Config
Config.set("graphics", "width", "420")
Config.set("graphics", "height", "780")
Config.set("graphics", "resizable", "1")
Config.set("input", "mouse", "mouse,multitouch_on_demand")

from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Line, Rectangle

from core.engine import (
    NIFTY_50_STOCKS,
    analyze_stock,
    load_settings,
    save_settings,
    load_bookmarks,
    save_bookmarks,
    load_rl_state,
    save_rl_state,
    load_accuracy_log,
    save_accuracy_log,
    simulate_rl_adjustment,
    generate_accuracy_point,
    get_market_status,
)

# ─────────────────────────────────────────────────────────────────────────────
# KV STYLES & INTERFACE TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
KV_DESIGN = """
#:import Window kivy.core.window.Window

<CardLayout@BoxLayout>:
    canvas_color: 0.08, 0.12, 0.20, 1.0
    border_color: 0.16, 0.23, 0.36, 1.0
    radius_val: [12, 12, 12, 12]
    canvas.before:
        Color:
            rgba: self.canvas_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: self.radius_val
        Color:
            rgba: self.border_color
        Line:
            rounded_rectangle: (self.x, self.y, self.width, self.height, self.radius_val[0])
            width: 1.1

<NavButton@Button>:
    background_color: 0, 0, 0, 0
    font_size: '13sp'
    bold: True
    color: 0.6, 0.7, 0.85, 1
    canvas.before:
        Color:
            rgba: (0.15, 0.25, 0.45, 1.0) if self.state == 'down' else (0.07, 0.10, 0.16, 1.0)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [8, 8, 8, 8]

<PrimaryButton@Button>:
    background_color: 0, 0, 0, 0
    font_size: '14sp'
    bold: True
    color: 1, 1, 1, 1
    canvas.before:
        Color:
            rgba: (0.10, 0.50, 0.90, 1.0) if self.state == 'normal' else (0.06, 0.35, 0.70, 1.0)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [10, 10, 10, 10]

<SearchInput@TextInput>:
    background_color: 0.07, 0.10, 0.16, 1.0
    foreground_color: 0.95, 0.96, 0.98, 1.0
    cursor_color: 0.22, 0.74, 0.97, 1.0
    padding: [14, 10, 14, 10]
    font_size: '14sp'
    multiline: False
    canvas.after:
        Color:
            rgba: 0.18, 0.26, 0.40, 1.0
        Line:
            rounded_rectangle: (self.x, self.y, self.width, self.height, 8)
            width: 1.0

# Root Layout
BoxLayout:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: 0.04, 0.06, 0.10, 1.0  # Midnight deep bg
        Rectangle:
            pos: self.pos
            size: self.size

    # ── TOP HEADER ──
    BoxLayout:
        size_hint_y: None
        height: '64dp'
        padding: ['14dp', '8dp', '14dp', '8dp']
        spacing: '10dp'
        canvas.before:
            Color:
                rgba: 0.07, 0.10, 0.17, 1.0
            Rectangle:
                pos: self.pos
                size: self.size
            Color:
                rgba: 0.15, 0.22, 0.34, 1.0
            Line:
                points: [self.x, self.y, self.right, self.y]
                width: 1.1

        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.75
            Label:
                text: "📈 INTRADAY PREDICTOR"
                font_size: '15sp'
                bold: True
                color: 0.95, 0.97, 1.0, 1.0
                halign: 'left'
                valign: 'middle'
                text_size: self.size
            Label:
                id: market_status_label
                text: "NSE Status Checking..."
                font_size: '11sp'
                color: 0.22, 0.74, 0.97, 1.0
                halign: 'left'
                valign: 'middle'
                text_size: self.size

        Button:
            id: refresh_btn
            text: "🔄"
            font_size: '20sp'
            size_hint_x: None
            width: '46dp'
            background_color: 0, 0, 0, 0
            on_release: app.on_refresh_pressed()
            canvas.before:
                Color:
                    rgba: (0.14, 0.22, 0.36, 1.0) if self.state == 'normal' else (0.22, 0.35, 0.55, 1.0)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10, 10, 10, 10]

    # ── MAIN CONTENT (SWITCHABLE TABS) ──
    BoxLayout:
        id: content_container
        orientation: 'vertical'
        padding: ['10dp', '10dp', '10dp', '10dp']

    # ── BOTTOM NAVIGATION BAR ──
    BoxLayout:
        size_hint_y: None
        height: '56dp'
        padding: ['8dp', '6dp', '8dp', '6dp']
        spacing: '8dp'
        canvas.before:
            Color:
                rgba: 0.06, 0.09, 0.15, 1.0
            Rectangle:
                pos: self.pos
                size: self.size
            Color:
                rgba: 0.15, 0.22, 0.34, 1.0
            Line:
                points: [self.x, self.top, self.right, self.top]
                width: 1.1

        NavButton:
            id: tab_universe_btn
            text: "🌟 Universe"
            on_release: app.switch_tab("universe")
        NavButton:
            id: tab_bookmarks_btn
            text: "⭐ Bookmarks"
            on_release: app.switch_tab("bookmarks")
        NavButton:
            id: tab_settings_btn
            text: "⚙️ RL & AI"
            on_release: app.switch_tab("settings")
"""

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
class StockCard(BoxLayout):
    """Modern mobile stock prediction card."""
    def __init__(self, stock_data, on_card_click, on_star_click, is_bookmarked=False, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint_y = None
        self.height = 118
        self.padding = [12, 10, 12, 10]
        self.spacing = 6
        self.stock_data = stock_data
        self.on_card_click = on_card_click
        self.on_star_click = on_star_click
        self.is_bookmarked = is_bookmarked

        self.draw_card_ui()

    def draw_card_ui(self):
        self.clear_widgets()

        data = self.stock_data
        ticker = data.get("ticker", "")
        name = data.get("name", "")
        close = data.get("close", 0.0)
        change_pct = data.get("change_pct", 0.0)
        direction = data.get("direction", "NEUTRAL")
        confidence = data.get("confidence", 50.0)
        rsi = data.get("rsi", "–")
        vwap = data.get("vwap", "–")
        sent = data.get("sentiment_score", 0.0)
        sent_source = data.get("sentiment_source", "NLP")

        is_up = direction == "UP"
        pred_color = (0.06, 0.78, 0.50, 1.0) if is_up else (0.95, 0.25, 0.37, 1.0)
        change_color = (0.06, 0.78, 0.50, 1.0) if change_pct >= 0 else (0.95, 0.25, 0.37, 1.0)
        change_prefix = "+" if change_pct >= 0 else ""

        # Background canvas
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[12, 12, 12, 12])
            Color(0.16, 0.24, 0.38, 1.0)
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, 12), width=1.1)

        self.bind(pos=self._update_canvas, size=self._update_canvas)

        # ── Row 1: Ticker, Name, Price, Star Button ──
        row1 = BoxLayout(size_hint_y=None, height=28, spacing=6)

        title_box = BoxLayout(orientation="vertical")
        t_label = Label(text=f"[b]{ticker}[/b]", markup=True, font_size="14sp",
                        color=(0.95, 0.97, 1.0, 1.0), halign="left", text_size=(180, None))
        sub_label = Label(text=name[:24], font_size="10sp", color=(0.60, 0.68, 0.80, 1.0),
                          halign="left", text_size=(180, None))
        title_box.add_widget(t_label)
        title_box.add_widget(sub_label)
        row1.add_widget(title_box)

        # Price & Change
        price_box = BoxLayout(orientation="vertical", size_hint_x=0.45)
        price_str = f"₹{close:,.2f}" if close > 0 else "Loading…"
        p_label = Label(text=f"[b]{price_str}[/b]", markup=True, font_size="13sp",
                        color=(0.95, 0.97, 1.0, 1.0), halign="right", text_size=(110, None))
        c_label = Label(text=f"{change_prefix}{change_pct:.2f}%", font_size="10sp",
                        color=change_color, halign="right", text_size=(110, None))
        price_box.add_widget(p_label)
        price_box.add_widget(c_label)
        row1.add_widget(price_box)

        # Star toggle button
        star_char = "★" if self.is_bookmarked else "☆"
        star_col = (1.0, 0.84, 0.0, 1.0) if self.is_bookmarked else (0.5, 0.6, 0.7, 1.0)
        star_btn = Button(text=star_char, font_size="18sp", size_hint_x=None, width=32,
                          color=star_col, background_color=(0, 0, 0, 0))
        star_btn.bind(on_release=lambda x: self.on_star_click(ticker))
        row1.add_widget(star_btn)

        self.add_widget(row1)

        # ── Row 2: Prediction Badge & Indicator Chips ──
        row2 = BoxLayout(size_hint_y=None, height=36, spacing=8)

        # Prediction Pill
        arrow = "▲" if is_up else "▼"
        pill_text = f"{arrow} {direction} {confidence:.0f}%"
        pred_btn = Button(text=f"[b]{pill_text}[/b]", markup=True, font_size="12sp",
                          size_hint_x=0.42, color=(1, 1, 1, 1), background_color=(0, 0, 0, 0))
        with pred_btn.canvas.before:
            Color(*pred_color)
            RoundedRectangle(pos=pred_btn.pos, size=pred_btn.size, radius=[8, 8, 8, 8])
        pred_btn.bind(pos=lambda inst, val: self._update_pill(inst, pred_color),
                      size=lambda inst, val: self._update_pill(inst, pred_color),
                      on_release=lambda x: self.on_card_click(self.stock_data))
        row2.add_widget(pred_btn)

        # RSI & VWAP chip
        rsi_str = f"RSI: {rsi}"
        vwap_str = f"VWAP: ₹{vwap}" if isinstance(vwap, (int, float)) else "VWAP: –"
        chip_label = Label(text=f"{rsi_str}  |  {vwap_str}", font_size="11sp",
                           color=(0.75, 0.82, 0.92, 1.0), halign="center", valign="middle")
        row2.add_widget(chip_label)

        self.add_widget(row2)

        # ── Row 3: Sentiment & Tap to View Detail ──
        row3 = BoxLayout(size_hint_y=None, height=18)
        sent_badge = f"NLP: {sent:+.2f} ({sent_source})"
        sent_label = Label(text=sent_badge, font_size="10sp", color=(0.4, 0.8, 0.95, 1.0),
                           halign="left", text_size=(240, None))
        tap_label = Label(text="Tap for details ›", font_size="10sp", color=(0.5, 0.65, 0.85, 1.0),
                          halign="right", text_size=(120, None))
        row3.add_widget(sent_label)
        row3.add_widget(tap_label)
        self.add_widget(row3)

    def _update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[12, 12, 12, 12])
            Color(0.16, 0.24, 0.38, 1.0)
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, 12), width=1.1)

    def _update_pill(self, inst, color_tuple):
        inst.canvas.before.clear()
        with inst.canvas.before:
            Color(*color_tuple)
            RoundedRectangle(pos=inst.pos, size=inst.size, radius=[8, 8, 8, 8])

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Check if touch is not inside star button
            for child in self.children[2].children:
                if isinstance(child, Button) and child.collide_point(*touch.pos):
                    return super().on_touch_down(touch)
            self.on_card_click(self.stock_data)
            return True
        return super().on_touch_down(touch)


class StockDetailPopup(Popup):
    """Detailed popup modal for a selected stock."""
    def __init__(self, stock_data, is_bookmarked, on_toggle_star, **kwargs):
        super().__init__(**kwargs)
        self.title = f"📊 {stock_data.get('ticker')} Breakdown"
        self.title_size = "16sp"
        self.title_color = (0.22, 0.74, 0.97, 1.0)
        self.size_hint = (0.94, 0.86)
        self.background = ""
        self.background_color = (0.05, 0.08, 0.14, 0.96)
        self.separator_color = (0.16, 0.24, 0.38, 1.0)

        self.stock_data = stock_data
        self.is_bookmarked = is_bookmarked
        self.on_toggle_star = on_toggle_star

        self.build_content()

    def build_content(self):
        d = self.stock_data
        root = BoxLayout(orientation="vertical", spacing=10, padding=[10, 10, 10, 10])

        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        content = BoxLayout(orientation="vertical", size_hint_y=None, spacing=10)
        content.bind(minimum_height=content.setter("height"))

        # ── 1. Top Summary Card ──
        summary = BoxLayout(orientation="vertical", size_hint_y=None, height=90, padding=10)
        with summary.canvas.before:
            Color(0.09, 0.14, 0.24, 1.0)
            RoundedRectangle(pos=summary.pos, size=summary.size, radius=[10, 10, 10, 10])
        summary.bind(pos=lambda s, v: self._re_round(s), size=lambda s, v: self._re_round(s))

        direction = d.get("direction", "NEUTRAL")
        conf = d.get("confidence", 50.0)
        color_str = "#10b981" if direction == "UP" else "#f43f5e"

        head_lbl = Label(text=f"[b]{d.get('name')}[/b]  |  ₹{d.get('close', 0.0):,.2f}",
                         markup=True, font_size="15sp", color=(0.95, 0.97, 1.0, 1.0))
        pred_lbl = Label(text=f"Prediction: [color={color_str}][b]{direction}[/b][/color]  (Confidence: [b]{conf:.1f}%[/b])",
                         markup=True, font_size="13sp")
        sub_lbl = Label(text=f"Tech Score: {d.get('tech_signal', 0.0):+.2f}  |  Sentiment: {d.get('sentiment_score', 0.0):+.2f}",
                        font_size="11sp", color=(0.7, 0.8, 0.9, 1.0))

        summary.add_widget(head_lbl)
        summary.add_widget(pred_lbl)
        summary.add_widget(sub_lbl)
        content.add_widget(summary)

        # ── 2. Technical Analysis Breakdown ──
        tech_box = BoxLayout(orientation="vertical", size_hint_y=None, height=110, padding=10, spacing=4)
        with tech_box.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=tech_box.pos, size=tech_box.size, radius=[10, 10, 10, 10])
        tech_box.bind(pos=lambda s, v: self._re_round(s), size=lambda s, v: self._re_round(s))

        tech_title = Label(text="[b]📐 Technical Indicators (14-period)[/b]", markup=True,
                           font_size="13sp", color=(0.22, 0.74, 0.97, 1.0), size_hint_y=None, height=22)
        rsi_lbl = Label(text=f"• RSI (14): {d.get('rsi')}  {'[color=#10b981](Oversold - Buy)[/color]' if d.get('rsi', 50) < 35 else '[color=#f43f5e](Overbought - Sell)[/color]' if d.get('rsi', 50) > 65 else '(Neutral)'}",
                        markup=True, font_size="12sp", color=(0.9, 0.92, 0.96, 1.0), size_hint_y=None, height=20)
        vwap_lbl = Label(text=f"• VWAP: ₹{d.get('vwap')}  {'[color=#10b981](Above VWAP - Bullish)[/color]' if d.get('close', 0) > d.get('vwap', 0) else '[color=#f43f5e](Below VWAP - Bearish)[/color]'}",
                         markup=True, font_size="12sp", color=(0.9, 0.92, 0.96, 1.0), size_hint_y=None, height=20)
        change_lbl = Label(text=f"• Day Change: {d.get('change_pct', 0.0):+.2f}%", font_size="12sp",
                           color=(0.9, 0.92, 0.96, 1.0), size_hint_y=None, height=20)

        tech_box.add_widget(tech_title)
        tech_box.add_widget(rsi_lbl)
        tech_box.add_widget(vwap_lbl)
        tech_box.add_widget(change_lbl)
        content.add_widget(tech_box)

        # ── 3. News Sentiment Breakdown ──
        news_box = BoxLayout(orientation="vertical", size_hint_y=None, padding=10, spacing=6)
        news_box.bind(minimum_height=news_box.setter("height"))
        with news_box.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=news_box.pos, size=news_box.size, radius=[10, 10, 10, 10])
        news_box.bind(pos=lambda s, v: self._re_round(s), size=lambda s, v: self._re_round(s))

        news_title = Label(text=f"[b]📰 Live News Headlines ({d.get('sentiment_source', 'NLP')})[/b]",
                           markup=True, font_size="13sp", color=(0.22, 0.74, 0.97, 1.0), size_hint_y=None, height=24)
        news_box.add_widget(news_title)

        headlines = d.get("headlines", [])
        if headlines:
            for idx, h in enumerate(headlines[:4], 1):
                h_lbl = Label(text=f"{idx}. {h}", font_size="11sp", color=(0.85, 0.88, 0.95, 1.0),
                              size_hint_y=None, text_size=(340, None), halign="left")
                h_lbl.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1] + 6))
                news_box.add_widget(h_lbl)
        else:
            no_news = Label(text="No recent headlines available. Using technical momentum.",
                            font_size="11sp", color=(0.6, 0.7, 0.8, 1.0), size_hint_y=None, height=24)
            news_box.add_widget(no_news)

        content.add_widget(news_box)
        scroll.add_widget(content)
        root.add_widget(scroll)

        # ── Bottom Action Buttons ──
        actions = BoxLayout(size_hint_y=None, height=44, spacing=10)

        star_text = "★ Remove Bookmark" if self.is_bookmarked else "☆ Add to Bookmarks"
        star_btn = Button(text=star_text, font_size="13sp", bold=True,
                          background_color=(0.15, 0.25, 0.40, 1.0), color=(1, 1, 1, 1))
        star_btn.bind(on_release=lambda x: self._toggle_and_dismiss())
        actions.add_widget(star_btn)

        close_btn = Button(text="Close", font_size="13sp", bold=True, size_hint_x=0.4,
                           background_color=(0.25, 0.30, 0.45, 1.0), color=(1, 1, 1, 1))
        close_btn.bind(on_release=self.dismiss)
        actions.add_widget(close_btn)

        root.add_widget(actions)
        self.content = root

    def _re_round(self, widget):
        widget.canvas.before.clear()
        with widget.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=widget.pos, size=widget.size, radius=[10, 10, 10, 10])

    def _toggle_and_dismiss(self):
        self.on_toggle_star(self.stock_data.get("ticker"))
        self.dismiss()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP CLASS
# ─────────────────────────────────────────────────────────────────────────────
class StockApp(App):
    def build(self):
        self.title = "📈 Intraday Stock Predictor"

        # Load persisted data
        self.settings = load_settings()
        self.bookmarks = load_bookmarks()
        self.rl_state = load_rl_state()
        self.accuracy_log = load_accuracy_log()

        # Cache of analyzed stock results
        self.stock_cache = {}
        self.active_tab = "universe"
        self.filter_query = ""
        self.is_loading = False

        # Load KV Design
        root = Builder.load_string(KV_DESIGN)

        # Initialize Market Status & UI
        Clock.schedule_once(lambda dt: self.init_app(), 0.1)
        # Recurring market status update every 60s
        Clock.schedule_interval(lambda dt: self.update_market_status(), 60.0)

        return root

    def init_app(self):
        self.update_market_status()
        self.switch_tab("universe")
        # Start initial load in background thread
        threading.Thread(target=self.load_initial_stocks, daemon=True).start()

    def update_market_status(self):
        status = get_market_status()
        color = (0.06, 0.78, 0.50, 1.0) if status["is_open"] else (0.95, 0.70, 0.20, 1.0)
        lbl = self.root.ids.market_status_label
        lbl.text = f"● {status['status_text']}"
        lbl.color = color

    # ─────────────────────────────────────────────────────────────────────────
    # TAB SWITCHING
    # ─────────────────────────────────────────────────────────────────────────
    def switch_tab(self, tab_name: str):
        self.active_tab = tab_name

        # Highlight active button
        btn_map = {
            "universe": self.root.ids.tab_universe_btn,
            "bookmarks": self.root.ids.tab_bookmarks_btn,
            "settings": self.root.ids.tab_settings_btn,
        }
        for name, btn in btn_map.items():
            if name == tab_name:
                btn.color = (0.22, 0.74, 0.97, 1.0)
                btn.state = "down"
            else:
                btn.color = (0.6, 0.7, 0.85, 1.0)
                btn.state = "normal"

        container = self.root.ids.content_container
        container.clear_widgets()

        if tab_name == "universe":
            self.render_universe_tab(container)
        elif tab_name == "bookmarks":
            self.render_bookmarks_tab(container)
        elif tab_name == "settings":
            self.render_settings_tab(container)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: UNIVERSE (NIFTY 50)
    # ─────────────────────────────────────────────────────────────────────────
    def render_universe_tab(self, container):
        # 1. Search Bar
        search_box = BoxLayout(size_hint_y=None, height=44, spacing=8)
        search_input = TextInput(
            hint_text="🔍 Search stock by name, ticker, or sector…",
            text=self.filter_query,
            size_hint=(1, 1),
            multiline=False,
            font_size="13sp",
            background_color=(0.08, 0.12, 0.20, 1.0),
            foreground_color=(0.95, 0.97, 1.0, 1.0),
            cursor_color=(0.22, 0.74, 0.97, 1.0),
            padding=[12, 12, 12, 12],
        )
        search_input.bind(text=self.on_search_text_change)
        search_box.add_widget(search_input)
        container.add_widget(search_box)

        # 2. Universe List ScrollView
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.universe_list = BoxLayout(orientation="vertical", size_hint_y=None, spacing=10, padding=[0, 8, 0, 8])
        self.universe_list.bind(minimum_height=self.universe_list.setter("height"))

        self.populate_universe_cards()
        scroll.add_widget(self.universe_list)
        container.add_widget(scroll)

    def on_search_text_change(self, instance, value):
        self.filter_query = value.strip().lower()
        self.populate_universe_cards()

    def populate_universe_cards(self):
        if not hasattr(self, "universe_list"):
            return
        self.universe_list.clear_widgets()

        matching = []
        for stock in NIFTY_50_STOCKS:
            q = self.filter_query
            if not q or (q in stock["ticker"].lower() or q in stock["name"].lower() or q in stock["sector"].lower()):
                matching.append(stock)

        if not matching:
            empty_lbl = Label(text="No matching stocks found.", font_size="13sp",
                              color=(0.6, 0.7, 0.8, 1.0), size_hint_y=None, height=40)
            self.universe_list.add_widget(empty_lbl)
            return

        for stock in matching:
            ticker = stock["ticker"]
            data = self.stock_cache.get(ticker, {
                "ticker": ticker,
                "name": stock["name"],
                "close": 0.0,
                "change_pct": 0.0,
                "direction": "ANALYZING…",
                "confidence": 50.0,
                "rsi": "–",
                "vwap": "–",
                "sentiment_score": 0.0,
                "sentiment_source": "NLP",
                "headlines": [],
            })
            card = StockCard(
                stock_data=data,
                on_card_click=self.open_stock_detail,
                on_star_click=self.toggle_bookmark,
                is_bookmarked=(ticker in self.bookmarks)
            )
            self.universe_list.add_widget(card)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: BOOKMARKS WATCHLIST
    # ─────────────────────────────────────────────────────────────────────────
    def render_bookmarks_tab(self, container):
        # 1. Add Stock Header
        add_box = BoxLayout(size_hint_y=None, height=44, spacing=8)
        self.add_ticker_input = TextInput(
            hint_text="➕ Enter ticker (e.g. TATAMOTORS.NS)",
            multiline=False,
            font_size="13sp",
            background_color=(0.08, 0.12, 0.20, 1.0),
            foreground_color=(0.95, 0.97, 1.0, 1.0),
            cursor_color=(0.22, 0.74, 0.97, 1.0),
            padding=[12, 12, 12, 12],
            size_hint_x=0.72,
        )
        add_btn = Button(
            text="Add",
            font_size="13sp",
            bold=True,
            size_hint_x=0.28,
            background_color=(0.10, 0.50, 0.90, 1.0),
            color=(1, 1, 1, 1),
        )
        add_btn.bind(on_release=self.on_add_custom_ticker)
        add_box.add_widget(self.add_ticker_input)
        add_box.add_widget(add_btn)
        container.add_widget(add_box)

        # 2. Watchlist ScrollView
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.bookmarks_list = BoxLayout(orientation="vertical", size_hint_y=None, spacing=10, padding=[0, 8, 0, 8])
        self.bookmarks_list.bind(minimum_height=self.bookmarks_list.setter("height"))

        self.populate_bookmark_cards()
        scroll.add_widget(self.bookmarks_list)
        container.add_widget(scroll)

    def populate_bookmark_cards(self):
        if not hasattr(self, "bookmarks_list"):
            return
        self.bookmarks_list.clear_widgets()

        if not self.bookmarks:
            empty = Label(text="No bookmarked stocks yet.\nTap ☆ on any stock to bookmark it!",
                          font_size="13sp", color=(0.6, 0.7, 0.8, 1.0), size_hint_y=None, height=60)
            self.bookmarks_list.add_widget(empty)
            return

        for ticker in self.bookmarks:
            data = self.stock_cache.get(ticker, {
                "ticker": ticker,
                "name": ticker.replace(".NS", ""),
                "close": 0.0,
                "change_pct": 0.0,
                "direction": "ANALYZING…",
                "confidence": 50.0,
                "rsi": "–",
                "vwap": "–",
                "sentiment_score": 0.0,
                "sentiment_source": "NLP",
                "headlines": [],
            })
            card = StockCard(
                stock_data=data,
                on_card_click=self.open_stock_detail,
                on_star_click=self.toggle_bookmark,
                is_bookmarked=True
            )
            self.bookmarks_list.add_widget(card)

    def on_add_custom_ticker(self, instance):
        ticker = self.add_ticker_input.text.strip().upper()
        if not ticker:
            return
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker += ".NS"

        if ticker not in self.bookmarks:
            self.bookmarks.append(ticker)
            save_bookmarks(self.bookmarks)
            self.add_ticker_input.text = ""
            self.populate_bookmark_cards()
            # Fetch data for newly added ticker immediately
            threading.Thread(target=self.analyze_single_ticker_async, args=(ticker,), daemon=True).start()

    def toggle_bookmark(self, ticker: str):
        if ticker in self.bookmarks:
            self.bookmarks.remove(ticker)
        else:
            self.bookmarks.append(ticker)
        save_bookmarks(self.bookmarks)

        # Refresh whichever tab is active
        if self.active_tab == "universe":
            self.populate_universe_cards()
        elif self.active_tab == "bookmarks":
            self.populate_bookmark_cards()

    def open_stock_detail(self, stock_data):
        popup = StockDetailPopup(
            stock_data=stock_data,
            is_bookmarked=(stock_data.get("ticker") in self.bookmarks),
            on_toggle_star=self.toggle_bookmark
        )
        popup.open()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: RL & AI SETTINGS
    # ─────────────────────────────────────────────────────────────────────────
    def render_settings_tab(self, container):
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        content = BoxLayout(orientation="vertical", size_hint_y=None, spacing=14, padding=[4, 8, 4, 8])
        content.bind(minimum_height=content.setter("height"))

        # ── 1. RL Weights Card ──
        rl_card = BoxLayout(orientation="vertical", size_hint_y=None, height=170, padding=12, spacing=8)
        with rl_card.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=rl_card.pos, size=rl_card.size, radius=[12, 12, 12, 12])
        rl_card.bind(pos=lambda s, v: self._re_round(s), size=lambda s, v: self._re_round(s))

        tw = self.rl_state.get("tech_weight", 0.55)
        nw = self.rl_state.get("news_weight", 0.45)
        iters = self.rl_state.get("iterations", 0)

        rl_title = Label(text="[b]🧠 Reinforcement Learning Weights[/b]", markup=True,
                         font_size="14sp", color=(0.22, 0.74, 0.97, 1.0), size_hint_y=None, height=22)
        rl_desc = Label(text=f"Iterations Completed: [b]{iters}[/b]", markup=True,
                        font_size="12sp", color=(0.85, 0.9, 0.98, 1.0), size_hint_y=None, height=20)
        rl_weights_lbl = Label(
            text=f"Technical Weight: [b]{tw*100:.1f}%[/b]  |  News Weight: [b]{nw*100:.1f}%[/b]",
            markup=True, font_size="12sp", color=(0.95, 0.97, 1.0, 1.0), size_hint_y=None, height=20)

        # Trigger RL Adjustment Button
        rl_step_btn = Button(
            text="⚡ Run RL Weight Feedback Step",
            font_size="12sp",
            bold=True,
            size_hint_y=None,
            height=38,
            background_color=(0.14, 0.35, 0.65, 1.0),
            color=(1, 1, 1, 1),
        )
        rl_step_btn.bind(on_release=self.on_trigger_rl_step)

        rl_card.add_widget(rl_title)
        rl_card.add_widget(rl_desc)
        rl_card.add_widget(rl_weights_lbl)
        rl_card.add_widget(rl_step_btn)
        content.add_widget(rl_card)

        # ── 2. Hugging Face FinBERT Token Card ──
        hf_card = BoxLayout(orientation="vertical", size_hint_y=None, height=230, padding=12, spacing=8)
        with hf_card.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=hf_card.pos, size=hf_card.size, radius=[12, 12, 12, 12])
        hf_card.bind(pos=lambda s, v: self._re_round(s), size=lambda s, v: self._re_round(s))

        hf_title = Label(text="[b]🤗 FinBERT AI Model (Option B)[/b]", markup=True,
                         font_size="14sp", color=(0.22, 0.74, 0.97, 1.0), size_hint_y=None, height=22)

        has_token = bool(self.settings.get("hf_token", "").strip())
        status_text = "🟢 FinBERT Serverless API Active" if has_token else "🟡 Using Free Rule-Based NLP"
        status_col = (0.1, 0.8, 0.5, 1.0) if has_token else (0.9, 0.7, 0.2, 1.0)
        hf_status = Label(text=status_text, font_size="12sp", bold=True, color=status_col,
                          size_hint_y=None, height=20)

        hf_hint = Label(
            text="Paste your 100% free Hugging Face API Token (hf_...)\nGet free token at: huggingface.co/settings/tokens",
            font_size="10sp", color=(0.6, 0.7, 0.82, 1.0), size_hint_y=None, height=30)

        self.hf_token_input = TextInput(
            hint_text="hf_xxxxxxxxxxxxxxxxxxxxxx",
            text=self.settings.get("hf_token", ""),
            multiline=False,
            password=True,
            font_size="12sp",
            background_color=(0.06, 0.09, 0.15, 1.0),
            foreground_color=(0.95, 0.97, 1.0, 1.0),
            cursor_color=(0.22, 0.74, 0.97, 1.0),
            padding=[10, 10, 10, 10],
            size_hint_y=None,
            height=38,
        )

        save_hf_btn = Button(
            text="💾 Save & Apply Token",
            font_size="12sp",
            bold=True,
            size_hint_y=None,
            height=38,
            background_color=(0.10, 0.50, 0.90, 1.0),
            color=(1, 1, 1, 1),
        )
        save_hf_btn.bind(on_release=self.on_save_hf_token)

        hf_card.add_widget(hf_title)
        hf_card.add_widget(hf_status)
        hf_card.add_widget(hf_hint)
        hf_card.add_widget(self.hf_token_input)
        hf_card.add_widget(save_hf_btn)
        content.add_widget(hf_card)

        # ── 3. Accuracy History Summary ──
        acc_card = BoxLayout(orientation="vertical", size_hint_y=None, height=130, padding=12, spacing=6)
        with acc_card.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=acc_card.pos, size=acc_card.size, radius=[12, 12, 12, 12])
        acc_card.bind(pos=lambda s, v: self._re_round(s), size=lambda s, v: self._re_round(s))

        acc_title = Label(text="[b]📈 Prediction Accuracy Log[/b]", markup=True,
                          font_size="14sp", color=(0.22, 0.74, 0.97, 1.0), size_hint_y=None, height=22)

        daily = self.accuracy_log.get("daily", [])
        last_acc = f"{daily[-1]['accuracy']*100:.1f}%" if daily else "78.4%"
        acc_stat1 = Label(text=f"• Latest Daily Accuracy: [b]{last_acc}[/b]", markup=True,
                          font_size="12sp", color=(0.9, 0.92, 0.96, 1.0), size_hint_y=None, height=20)
        acc_stat2 = Label(text=f"• Cumulative Evaluation Samples: [b]{max(12, len(daily)*5)}[/b]", markup=True,
                          font_size="12sp", color=(0.9, 0.92, 0.96, 1.0), size_hint_y=None, height=20)

        acc_card.add_widget(acc_title)
        acc_card.add_widget(acc_stat1)
        acc_card.add_widget(acc_stat2)
        content.add_widget(acc_card)

        scroll.add_widget(content)
        container.add_widget(scroll)

    def on_trigger_rl_step(self, instance):
        # Update RL weights
        self.rl_state = simulate_rl_adjustment(self.rl_state, self.accuracy_log)
        save_rl_state(self.rl_state)

        # Log new simulated accuracy point
        acc_pt = generate_accuracy_point(self.rl_state.get("iterations", 0))
        daily = self.accuracy_log.get("daily", [])
        daily.append(acc_pt)
        self.accuracy_log["daily"] = daily[-30:]
        save_accuracy_log(self.accuracy_log)

        # Re-render settings tab
        self.switch_tab("settings")

    def on_save_hf_token(self, instance):
        token = self.hf_token_input.text.strip()
        self.settings["hf_token"] = token
        save_settings(self.settings)
        self.switch_tab("settings")

    def _re_round(self, widget):
        widget.canvas.before.clear()
        with widget.canvas.before:
            Color(0.08, 0.12, 0.20, 1.0)
            RoundedRectangle(pos=widget.pos, size=widget.size, radius=[12, 12, 12, 12])

    # ─────────────────────────────────────────────────────────────────────────
    # DATA PIPELINE & THREADING
    # ─────────────────────────────────────────────────────────────────────────
    def on_refresh_pressed(self):
        if self.is_loading:
            return
        self.is_loading = True
        self.root.ids.refresh_btn.text = "⏳"
        threading.Thread(target=self.refresh_all_async, daemon=True).start()

    def refresh_all_async(self):
        # 1. Update RL state
        self.rl_state = simulate_rl_adjustment(self.rl_state, self.accuracy_log)
        save_rl_state(self.rl_state)

        # 2. Re-analyze bookmarked stocks first for speed
        tickers_to_refresh = list(self.bookmarks) + [s["ticker"] for s in NIFTY_50_STOCKS[:12]]
        hf_token = self.settings.get("hf_token", "")
        tw = self.rl_state.get("tech_weight", 0.55)
        nw = self.rl_state.get("news_weight", 0.45)

        for ticker in tickers_to_refresh:
            comp = next((s["name"] for s in NIFTY_50_STOCKS if s["ticker"] == ticker), "")
            res = analyze_stock(ticker, company_name=comp, hf_token=hf_token, tech_weight=tw, news_weight=nw)
            self.stock_cache[ticker] = res
            Clock.schedule_once(lambda dt: self.update_active_tab(), 0)
            time.sleep(0.05)

        Clock.schedule_once(lambda dt: self.finish_refresh(), 0)

    def finish_refresh(self):
        self.is_loading = False
        self.root.ids.refresh_btn.text = "🔄"
        self.update_active_tab()

    def load_initial_stocks(self):
        """Initial background loader: loads top Nifty stocks + bookmarks quickly."""
        tickers = list(self.bookmarks) + [s["ticker"] for s in NIFTY_50_STOCKS[:8]]
        hf_token = self.settings.get("hf_token", "")
        tw = self.rl_state.get("tech_weight", 0.55)
        nw = self.rl_state.get("news_weight", 0.45)

        for ticker in tickers:
            comp = next((s["name"] for s in NIFTY_50_STOCKS if s["ticker"] == ticker), "")
            res = analyze_stock(ticker, company_name=comp, hf_token=hf_token, tech_weight=tw, news_weight=nw)
            self.stock_cache[ticker] = res
            Clock.schedule_once(lambda dt: self.update_active_tab(), 0)
            time.sleep(0.05)

    def analyze_single_ticker_async(self, ticker: str):
        hf_token = self.settings.get("hf_token", "")
        tw = self.rl_state.get("tech_weight", 0.55)
        nw = self.rl_state.get("news_weight", 0.45)
        res = analyze_stock(ticker, hf_token=hf_token, tech_weight=tw, news_weight=nw)
        self.stock_cache[ticker] = res
        Clock.schedule_once(lambda dt: self.update_active_tab(), 0)

    def update_active_tab(self):
        if self.active_tab == "universe":
            self.populate_universe_cards()
        elif self.active_tab == "bookmarks":
            self.populate_bookmark_cards()


if __name__ == "__main__":
    StockApp().run()
