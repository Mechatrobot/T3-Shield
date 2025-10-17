import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QFont, QPainter, QPainterPath, QPixmap, QColor, QPen, QBrush
from PyQt5.QtCore import Qt, QTimer, QDateTime, QRect

class RoundedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333333;
                border: none;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                text-align: left;
                padding-left: 80px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """)

class CircularButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(60, 60)
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 59, 48, 0.8);
                color: white;
                border: none;
                border-radius: 30px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 59, 48, 1);
            }
            QPushButton:pressed {
                background-color: rgba(200, 40, 30, 1);
            }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analysis System")
        self.setGeometry(int(500), int(500), 800, 480)
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Initialize gradient animation variables
        self.gradient_offset = 0.0
        
        # Set initial background gradient (green and white)
        self.update_background_gradient()
        
        self.UI()
        
        # Update time every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.timeout.connect(self.update_wifi_icon)
        self.timer.timeout.connect(self.update_battery_icon)
        self.timer.start(1000)
        
        # Separate timer for faster background animation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_background)
        self.animation_timer.start(100)  # Update every 100ms for smoother/faster animation
        
    def UI(self):
        # Logo placeholder (top left) - space for image without border (larger size)
        self.logo_label = QLabel(self)
        self.logo_label.setGeometry(15, 3, 60, 60)
        self.logo_label.setStyleSheet("background-color: transparent; border-radius: 5px;")
        self.logo_label.setAlignment(Qt.AlignCenter)
        # Uncomment and use this to load your logo image:
        logo_pixmap = QPixmap("C:/Users/HP/Downloads/logo.png")
        self.logo_label.setPixmap(logo_pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # WiFi icon (iOS style - black) - Custom drawn
        self.wifi_icon = QLabel(self)
        self.wifi_icon.setGeometry(570, 20, 25, 25)
        self.wifi_icon.setStyleSheet("background: transparent;")
        self.wifi_icon.setAlignment(Qt.AlignCenter)
        self.wifi_strength = 100  # Initialize
        self.update_wifi_icon()
        
        # Battery icon (iOS style - black) - Custom drawn
        self.battery_icon = QLabel(self)
        self.battery_icon.setGeometry(625, 20, 35, 20)
        self.battery_icon.setStyleSheet("background: transparent;")
        self.battery_icon.setAlignment(Qt.AlignCenter)
        self.battery_level = 100  # Initialize
        self.is_charging = False
        self.update_battery_icon()
        
        # Settings icon (iOS style gear - black)
        settings_icon = QPushButton("⚙", self)
        settings_icon.setGeometry(685, 17, 30, 30)
        settings_icon.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #333333;
                border: none;
                font-size: 20px;
            }
            QPushButton:hover {
                color: #007AFF;
            }
        """)
        settings_icon.clicked.connect(self.open_settings)
        
        # Date and time (iOS style - black text)
        self.datetime_label = QLabel(self)
        self.datetime_label.setGeometry(740, 17, 50, 30)
        self.datetime_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.datetime_label.setStyleSheet("color: #333333; background: transparent;")
        self.datetime_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.update_time()
        
        # Main buttons horizontally aligned in the center
        button_width = 220
        button_height = 70
        button_spacing = 25
        center_y = 200  # Moved up from 240 to better center vertically
        
        # Calculate starting X position to center all three buttons
        total_width = 3 * button_width + 2 * button_spacing
        start_x = (800 - total_width) // 2
        
        # General Analysis Button
        self.btn_general = RoundedButton("General\nAnalysis", self)
        self.btn_general.setGeometry(start_x, center_y, button_width, button_height)
        
        # Image placeholder for General Analysis button (left side of button)
        self.img_general = QLabel(self.btn_general)
        self.img_general.setGeometry(15, 10, 50, 50)
        self.img_general.setStyleSheet("background-color: transparent; border: none;")
        self.img_general.setAlignment(Qt.AlignCenter)
        # Uncomment to load your image:
        img_pixmap = QPixmap("C:/Users/HP/Downloads/Group.png")
        self.img_general.setPixmap(img_pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Individual Analysis Button
        self.btn_individual = RoundedButton("Individual\nAnalysis", self)
        self.btn_individual.setGeometry(start_x + button_width + button_spacing, center_y, 
                                        button_width, button_height)
        
        # Image placeholder for Individual Analysis button
        self.img_individual = QLabel(self.btn_individual)
        self.img_individual.setGeometry(15, 10, 50, 50)
        self.img_individual.setStyleSheet("background-color: transparent; border: none;")
        self.img_individual.setAlignment(Qt.AlignCenter)
        # Uncomment to load your image:
        img_pixmap = QPixmap("C:/Users/HP/Downloads/single.png")
        self.img_individual.setPixmap(img_pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Hardware Test Button
        self.btn_hardware = RoundedButton("Hardware\nTest", self)
        self.btn_hardware.setGeometry(start_x + 2 * (button_width + button_spacing), center_y, 
                                      button_width, button_height)
        
        # Image placeholder for Hardware Test button
        self.img_hardware = QLabel(self.btn_hardware)
        self.img_hardware.setGeometry(15, 10, 50, 50)
        self.img_hardware.setStyleSheet("background-color: transparent; border: none;")
        self.img_hardware.setAlignment(Qt.AlignCenter)
        # Uncomment to load your image:
        img_pixmap = QPixmap("C:/Users/HP/Downloads/Hardware.png")
        self.img_hardware.setPixmap(img_pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Version label (bottom left)
        version_label = QLabel("v1.0.0", self)
        version_label.setGeometry(20, 440, 100, 30)
        version_label.setFont(QFont("Arial", 10))
        version_label.setStyleSheet("color: rgba(100, 100, 100, 0.8); background: transparent;")
        
        # Copyright label (bottom center)
        copyright_label = QLabel("@2025. Sensthings.", self)
        copyright_label.setGeometry(300, 440, 200, 30)
        copyright_label.setFont(QFont("Arial", 10))
        copyright_label.setStyleSheet("color: rgba(100, 100, 100, 0.8); background: transparent;")
        copyright_label.setAlignment(Qt.AlignCenter)
        
        # Help button (circular, bottom right - before shutdown)
        btn_help = CircularButton("?", self)
        btn_help.setGeometry(650, 405, 60, 60)
        btn_help.setStyleSheet("""
            QPushButton {
                background-color: rgba(52, 152, 219, 0.8);
                color: white;
                border: none;
                border-radius: 30px;
                font-size: 28px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 1);
            }
            QPushButton:pressed {
                background-color: rgba(41, 128, 185, 1);
            }
        """)
        btn_help.clicked.connect(self.show_help)
        
        # Circular Shutdown button (bottom right)
        btn_shutdown = CircularButton("", self)  # Empty text
        btn_shutdown.setGeometry(720, 405, 60, 60)
        btn_shutdown.clicked.connect(self.shutdown)
        
        # Add shutdown icon inside the button
        shutdown_icon = QLabel(btn_shutdown)
        shutdown_icon.setGeometry(10, 10, 40, 40)  # Center the icon
        shutdown_icon.setStyleSheet("background: transparent;")
        shutdown_icon.setAlignment(Qt.AlignCenter)
        # Option 1: Use an image file
        shutdown_pixmap = QPixmap("C:/Users/HP/Downloads/OFF.png")
        shutdown_icon.setPixmap(shutdown_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def update_time(self):
        current_time = QDateTime.currentDateTime()
        time_text = current_time.toString("hh:mm")
        self.datetime_label.setText(time_text)
    
    def update_background_gradient(self):
        """Update the background gradient with animated offset"""
        # Calculate gradient positions based on offset
        x1 = self.gradient_offset
        y1 = 0
        x2 = 1 - self.gradient_offset
        y2 = 1
        
        self.setStyleSheet(f"""
            QMainWindow {{
                background: qlineargradient(x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2},
                    stop:0 #a8e6cf, stop:0.5 #dcedc1, stop:1 #ffffff);
            }}
        """)
    
    def animate_background(self):
        """Animate the background gradient by shifting it"""
        # Increment offset (creates the moving effect) - faster increment
        self.gradient_offset += 0.02
        
        # Reset offset when it reaches 1 to create continuous loop
        if self.gradient_offset >= 1.0:
            self.gradient_offset = 0.0
        
        # Update the gradient
        self.update_background_gradient()
    
    def draw_wifi_icon(self, strength):
        """Draw WiFi icon with signal bars based on strength (0-100)"""
        pixmap = QPixmap(25, 25)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # WiFi color (black/dark gray)
        color = QColor(51, 51, 51)
        painter.setPen(Qt.NoPen)
        
        # Draw WiFi arcs (3 bars)
        # Determine how many bars to show
        bars = 0
        if strength >= 75:
            bars = 3
        elif strength >= 50:
            bars = 2
        elif strength >= 25:
            bars = 1
        
        # Center point
        center_x, center_y = 12, 20
        
        # Draw 3 arcs with different opacities based on signal
        arc_sizes = [(6, 6), (10, 10), (14, 14)]
        
        for i, (w, h) in enumerate(arc_sizes):
            if i < bars:
                painter.setBrush(QBrush(color))
            else:
                # Dim color for inactive bars
                dim_color = QColor(51, 51, 51, 50)
                painter.setBrush(QBrush(dim_color))
            
            # Draw arc segment
            rect = QRect(center_x - w, center_y - h, w * 2, h * 2)
            painter.drawPie(rect, 30 * 16, 120 * 16)  # Arc from 30° to 150°
        
        # Draw center dot
        painter.setBrush(QBrush(color))
        painter.drawEllipse(center_x - 1, center_y - 1, 2, 2)
        
        painter.end()
        return pixmap
    
    def draw_battery_icon(self, level, charging=False):
        """Draw battery icon with level indicator (0-100)"""
        pixmap = QPixmap(35, 20)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Battery outline (black/dark gray)
        outline_color = QColor(51, 51, 51)
        painter.setPen(QPen(outline_color, 2))
        painter.setBrush(Qt.NoBrush)
        
        # Draw battery body
        battery_body = QRect(2, 4, 26, 12)
        painter.drawRoundedRect(battery_body, 2, 2)
        
        # Draw battery tip
        painter.setBrush(QBrush(outline_color))
        battery_tip = QRect(28, 7, 4, 6)
        painter.drawRect(battery_tip)
        
        # Draw battery fill based on level
        if level > 20:
            fill_color = QColor(51, 51, 51)
        else:
            fill_color = QColor(255, 59, 48)  # Red for low battery
        
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)
        
        # Calculate fill width (leave 2px margin)
        fill_width = int((level / 100.0) * 22)
        if fill_width > 0:
            battery_fill = QRect(4, 6, fill_width, 8)
            painter.drawRoundedRect(battery_fill, 1, 1)
        
        # Draw charging bolt if charging
        if charging:
            painter.setBrush(QBrush(QColor(255, 204, 0)))  # Yellow bolt
            painter.setPen(Qt.NoPen)
            
            # Simple lightning bolt shape
            painter.drawEllipse(13, 8, 4, 4)
        
        painter.end()
        return pixmap
    
    def update_wifi_icon(self):
        # Simulate WiFi strength (0-100)
        # Replace with actual WiFi detection
        import random
        self.wifi_strength = random.randint(0, 100)
        
        # Draw and set the WiFi icon
        pixmap = self.draw_wifi_icon(self.wifi_strength)
        self.wifi_icon.setPixmap(pixmap)
        
        # To get actual WiFi signal on Windows:
        # import subprocess
        # result = subprocess.check_output(['netsh', 'wlan', 'show', 'interfaces']).decode('utf-8')
        # Parse for signal strength
    
    def update_battery_icon(self):
        # Simulate battery level (0-100)
        # Replace with actual battery detection using psutil
        import random
        self.battery_level = random.randint(0, 100)
        self.is_charging = random.choice([True, False])
        
        # Draw and set the battery icon
        pixmap = self.draw_battery_icon(self.battery_level, self.is_charging)
        self.battery_icon.setPixmap(pixmap)
        
        # To get actual battery level:
        # import psutil
        # battery = psutil.sensors_battery()
        # if battery:
        #     self.battery_level = battery.percent
        #     self.is_charging = battery.power_plugged
    
    def open_settings(self):
        print("Settings button clicked!")
        # Add your settings functionality here
    
    def show_help(self):
        print("Help button clicked!")
        # Add your help functionality here
    
    def shutdown(self):
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())