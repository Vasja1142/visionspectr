import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QScrollArea, QLabel,
    QVBoxLayout, QHBoxLayout, QGroupBox, QSlider, QSpinBox,
    QPushButton, QFileDialog, QStatusBar, QWidget, QMessageBox,
    QRadioButton, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPen, QPainter, QColor


# ============================================================
# Функция плавного склеивания двух изображений
# ============================================================
def blend_images_smooth(left_img, right_img, blend_width=100):
    h_min = min(left_img.shape[0], right_img.shape[0])
    left_img = left_img[:h_min, :]
    right_img = right_img[:h_min, :]

    h, w_left = left_img.shape[:2]
    _, w_right = right_img.shape[:2]

    blend_width = min(blend_width, w_left, w_right)

    mask = np.ones((h, w_left), dtype=np.float32)
    for i in range(blend_width):
        alpha = 1.0 - (i / blend_width)
        mask[:, w_left - blend_width + i] = alpha

    if len(left_img.shape) == 3:
        mask = np.stack([mask] * 3, axis=-1)

    total_width = w_left + w_right - blend_width
    result = np.zeros((h, total_width, left_img.shape[2]), dtype=np.uint8)

    result[:, :w_left - blend_width] = left_img[:, :w_left - blend_width]
    result[:, w_left:] = right_img[:, blend_width:]

    left_overlap = left_img[:, w_left - blend_width:].astype(np.float32)
    right_overlap = right_img[:, :blend_width].astype(np.float32)
    mask_overlap = mask[:, w_left - blend_width:].astype(np.float32)

    blended_overlap = left_overlap * mask_overlap + right_overlap * (1.0 - mask_overlap)
    result[:, w_left - blend_width:w_left] = blended_overlap.astype(np.uint8)

    return result


# ============================================================
# Кастомный QLabel для кликов по точкам
# ============================================================
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; color: #aaaaaa;")
        self.points = []
        self.max_points = 4
        self.on_point_added = None
        self.original_pixmap = None

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_display()

    def update_display(self):
        if self.original_pixmap is None:
            return

        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)

        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                  QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]

        for i, (x, y) in enumerate(self.points):
            color = colors[i % len(colors)]
            painter.setPen(QPen(color, 3))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 128))
            painter.drawEllipse(int(x) - 8, int(y) - 8, 16, 16)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawText(int(x) + 10, int(y) - 10, str(i + 1))

        painter.end()
        super().setPixmap(pixmap)

    def mousePressEvent(self, event):
        if len(self.points) < self.max_points and self.original_pixmap is not None:
            pixmap = self.original_pixmap
            label_size = self.size()
            pixmap_size = pixmap.size()

            scale_x = pixmap_size.width() / label_size.width()
            scale_y = pixmap_size.height() / label_size.height()

            # ИСПРАВЛЕНО: используем position() вместо pos()
            x = event.position().x() * scale_x
            y = event.position().y() * scale_y

            self.points.append((x, y))
            self.update_display()

            if self.on_point_added:
                self.on_point_added(len(self.points))

    def clear_points(self):
        self.points = []
        self.update_display()


# ============================================================
# Главное окно
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionSpectr MVP - Выравнивание паллеты")
        self.resize(1200, 700)

        self.original_img = None
        self.current_img = None

        self._init_ui()

    def _init_ui(self):
        # Меню
        file_menu = self.menuBar().addMenu("Файл")
        open_action = file_menu.addAction("Открыть изображение...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)

        save_action = file_menu.addAction("Сохранить как...")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)

        self.menuBar().addMenu("Справка").addAction("О программе").triggered.connect(
            lambda: QMessageBox.information(
                self, "О программе", "VisionSpectr MVP v0.1\nВыравнивание паллеты"
            )
        )

        # Центральная область
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Левая часть: изображение
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = ImageLabel()
        self.scroll_area.setWidget(self.image_label)
        splitter.addWidget(self.scroll_area)

        # Правая часть: параметры
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Режим выбора точек
        mode_group = QGroupBox("Режим выбора точек")
        mode_layout = QVBoxLayout()

        self.mode_4points = QRadioButton("4 точки (обычная перспектива)")
        self.mode_6points = QRadioButton("6 точек (двойная перспектива)")
        self.mode_4points.setChecked(True)

        self.mode_4points.toggled.connect(self.on_mode_changed)
        self.mode_6points.toggled.connect(self.on_mode_changed)

        mode_layout.addWidget(self.mode_4points)
        mode_layout.addWidget(self.mode_6points)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        # Статус выбора точек
        self.status_label = QLabel("Откройте изображение и кликните по углам")
        right_layout.addWidget(self.status_label)

        self.point_counter = QLabel("Точек выбрано: 0/4")
        right_layout.addWidget(self.point_counter)

        # Кнопки управления точками
        self.btn_clear_points = QPushButton("Очистить точки")
        self.btn_clear_points.clicked.connect(self.clear_points)
        right_layout.addWidget(self.btn_clear_points)

        # Кнопка применения перспективы
        self.btn_apply_perspective = QPushButton("Применить перспективу")
        self.btn_apply_perspective.clicked.connect(self.apply_perspective)
        self.btn_apply_perspective.setEnabled(False)
        right_layout.addWidget(self.btn_apply_perspective)

        # Поворот
        rot_group = QGroupBox("Поворот (градусы)")
        rot_layout = QHBoxLayout()
        self.rot_slider = QSlider(Qt.Orientation.Horizontal)
        self.rot_slider.setRange(-180, 180)
        self.rot_spin = QSpinBox()
        self.rot_spin.setRange(-180, 180)
        self.rot_slider.valueChanged.connect(self.rot_spin.setValue)
        self.rot_spin.valueChanged.connect(self.rot_slider.setValue)
        self.rot_slider.valueChanged.connect(self.process_image)
        rot_layout.addWidget(self.rot_slider)
        rot_layout.addWidget(self.rot_spin)
        rot_group.setLayout(rot_layout)
        right_layout.addWidget(rot_group)

        # Масштаб
        scale_group = QGroupBox("Масштаб (%)")
        scale_layout = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 300)
        self.scale_slider.setValue(100)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(10, 300)
        self.scale_spin.setValue(100)
        self.scale_slider.valueChanged.connect(self.scale_spin.setValue)
        self.scale_spin.valueChanged.connect(self.scale_slider.setValue)
        self.scale_slider.valueChanged.connect(self.process_image)
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_spin)
        scale_group.setLayout(scale_layout)
        right_layout.addWidget(scale_group)

        # Размер изображения
        size_group = QGroupBox("Размер изображения (пиксели)")
        size_layout = QVBoxLayout()

        self.current_size_label = QLabel("Текущий размер: не загружено")
        size_layout.addWidget(self.current_size_label)

        size_input_layout = QHBoxLayout()
        size_input_layout.addWidget(QLabel("Ширина:"))
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("Ширина")
        self.width_input.editingFinished.connect(self.on_size_changed)
        size_input_layout.addWidget(self.width_input)

        size_input_layout.addWidget(QLabel("Высота:"))
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Высота")
        self.height_input.editingFinished.connect(self.on_size_changed)
        size_input_layout.addWidget(self.height_input)

        size_layout.addLayout(size_input_layout)

        self.keep_aspect = QCheckBox("Сохранять пропорции")
        self.keep_aspect.setChecked(True)
        size_layout.addWidget(self.keep_aspect)

        size_btn_layout = QHBoxLayout()
        self.btn_apply_size = QPushButton("Применить размер")
        self.btn_apply_size.clicked.connect(self.apply_size)
        size_btn_layout.addWidget(self.btn_apply_size)

        self.btn_reset_size = QPushButton("Сбросить размер")
        self.btn_reset_size.clicked.connect(self.reset_size)
        size_btn_layout.addWidget(self.btn_reset_size)

        size_layout.addLayout(size_btn_layout)
        size_group.setLayout(size_layout)
        right_layout.addWidget(size_group)

        # Кнопки внизу
        right_layout.addStretch()
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Сбросить всё")
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_apply = QPushButton("Сохранить результат")
        self.btn_apply.clicked.connect(self.save_image)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_apply)
        right_layout.addLayout(btn_layout)

        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)
        splitter.setSizes([800, 400])

        self.image_label.on_point_added = self.update_point_counter

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

    def on_mode_changed(self):
        self.clear_points()
        if self.mode_4points.isChecked():
            self.image_label.max_points = 4
            self.status_label.setText("Режим: 4 точки. Кликните по 4 углам объекта")
        else:
            self.image_label.max_points = 6
            self.status_label.setText("Режим: 6 точек. Кликните: верх-лево, верх-центр, верх-право, низ-лево, низ-центр, низ-право")

    def update_point_counter(self, count):
        max_pts = self.image_label.max_points
        self.point_counter.setText(f"Точек выбрано: {count}/{max_pts}")
        if count == max_pts:
            self.status_label.setText(f"Все {max_pts} точки выбраны. Нажмите 'Применить перспективу'")
            self.btn_apply_perspective.setEnabled(True)
        else:
            if max_pts == 4:
                self.status_label.setText("Кликните по следующему углу")
            else:
                labels = ["верх-лево", "верх-центр", "верх-право", "низ-лево", "низ-центр", "низ-право"]
                if count < len(labels):
                    self.status_label.setText(f"Следующая точка: {labels[count]}")

    def clear_points(self):
        self.image_label.clear_points()
        max_pts = self.image_label.max_points
        self.update_point_counter(0)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            self.status_bar.showMessage("Ошибка чтения файла")
            return

        self.original_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.current_img = self.original_img.copy()

        h, w, ch = self.original_img.shape
        bytes_per_line = ch * w
        q_img = QImage(
            self.original_img.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.set_image(pixmap)

        self.width_input.setText(str(w))
        self.height_input.setText(str(h))
        self.current_size_label.setText(f"Текущий размер: {w}x{h}")

        self.clear_points()
        self.reset_params(silent=True)
        self.status_bar.showMessage(f"Загружено: {file_path} | Размер: {w}x{h}")

    def on_size_changed(self):
        if not self.keep_aspect.isChecked() or self.original_img is None:
            return

        h, w = self.original_img.shape[:2]
        aspect_ratio = w / h

        sender = self.sender()
        if sender == self.width_input:
            try:
                new_width = int(self.width_input.text())
                new_height = int(new_width / aspect_ratio)
                self.height_input.blockSignals(True)
                self.height_input.setText(str(new_height))
                self.height_input.blockSignals(False)
            except ValueError:
                pass
        elif sender == self.height_input:
            try:
                new_height = int(self.height_input.text())
                new_width = int(new_height * aspect_ratio)
                self.width_input.blockSignals(True)
                self.width_input.setText(str(new_width))
                self.width_input.blockSignals(False)
            except ValueError:
                pass

    def apply_size(self):
        if self.original_img is None:
            QMessageBox.warning(self, "Ошибка", "Сначала откройте изображение")
            return

        try:
            new_width = int(self.width_input.text())
            new_height = int(self.height_input.text())
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные размеры")
            return

        if new_width <= 0 or new_height <= 0:
            QMessageBox.warning(self, "Ошибка", "Размеры должны быть больше 0")
            return

        resized = cv2.resize(
            self.original_img, (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        self.original_img = resized
        self.current_img = resized.copy()

        self.update_display()
        self.status_bar.showMessage(f"Размер изменён на: {new_width}x{new_height}")

    def reset_size(self):
        if self.original_img is None:
            return

        h, w = self.original_img.shape[:2]
        self.width_input.setText(str(w))
        self.height_input.setText(str(h))
        self.status_bar.showMessage("Размер сброшен к текущему")

    def apply_perspective(self):
        max_pts = self.image_label.max_points
        if len(self.image_label.points) != max_pts:
            QMessageBox.warning(self, "Ошибка", f"Выберите ровно {max_pts} точки")
            return

        if max_pts == 4:
            self.apply_4point_perspective()
        else:
            self.apply_6point_perspective()

    def apply_4point_perspective(self):
        src_points = np.array(self.image_label.points, dtype=np.float32)

        width_top = np.linalg.norm(src_points[1] - src_points[0])
        width_bottom = np.linalg.norm(src_points[2] - src_points[3])
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(src_points[3] - src_points[0])
        height_right = np.linalg.norm(src_points[2] - src_points[1])
        max_height = int(max(height_left, height_right))

        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(
            self.original_img, matrix, (max_width, max_height)
        )

        self.original_img = transformed
        self.current_img = transformed.copy()

        h, w = self.original_img.shape[:2]
        self.width_input.setText(str(w))
        self.height_input.setText(str(h))
        self.current_size_label.setText(f"Текущий размер: {w}x{h}")

        self.update_display()
        self.clear_points()
        self.status_bar.showMessage(f"Перспектива применена. Новый размер: {max_width}x{max_height}")

    def apply_6point_perspective(self):
        pts = np.array(self.image_label.points, dtype=np.float32)

        src_left = np.array([pts[0], pts[1], pts[4], pts[3]], dtype=np.float32)
        src_right = np.array([pts[1], pts[2], pts[5], pts[4]], dtype=np.float32)

        width_left_top = np.linalg.norm(pts[1] - pts[0])
        width_left_bottom = np.linalg.norm(pts[4] - pts[3])
        max_width_left = int(max(width_left_top, width_left_bottom))

        width_right_top = np.linalg.norm(pts[2] - pts[1])
        width_right_bottom = np.linalg.norm(pts[5] - pts[4])
        max_width_right = int(max(width_right_top, width_right_bottom))

        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[4] - pts[1])
        height_right2 = np.linalg.norm(pts[5] - pts[2])
        max_height = int(max(height_left, height_right, height_right2))

        dst_left = np.array([
            [0, 0],
            [max_width_left - 1, 0],
            [max_width_left - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        dst_right = np.array([
            [0, 0],
            [max_width_right - 1, 0],
            [max_width_right - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix_left = cv2.getPerspectiveTransform(src_left, dst_left)
        matrix_right = cv2.getPerspectiveTransform(src_right, dst_right)

        warped_left = cv2.warpPerspective(
            self.original_img, matrix_left, (max_width_left, max_height)
        )
        warped_right = cv2.warpPerspective(
            self.original_img, matrix_right, (max_width_right, max_height)
        )

        self.current_img = blend_images_smooth(warped_left, warped_right, blend_width=100)
        self.original_img = self.current_img.copy()

        h, w = self.original_img.shape[:2]
        self.width_input.setText(str(w))
        self.height_input.setText(str(h))
        self.current_size_label.setText(f"Текущий размер: {w}x{h}")

        self.update_display()
        self.clear_points()
        self.status_bar.showMessage("Двойная перспектива применена с плавным швом")

    def process_image(self):
        if self.original_img is None:
            return

        angle = self.rot_slider.value()
        scale = self.scale_slider.value() / 100.0

        h, w = self.original_img.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        transformed = cv2.warpAffine(
            self.original_img, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        self.current_img = transformed
        self.update_display()

    def update_display(self):
        if self.current_img is None:
            return

        h, w, ch = self.current_img.shape
        bytes_per_line = ch * w
        q_img = QImage(
            self.current_img.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.set_image(pixmap)
        self.status_bar.showMessage(
            f"Размер: {w}x{h} | "
            f"Поворот: {self.rot_slider.value()}° | "
            f"Масштаб: {self.scale_slider.value()}%"
        )

    def reset_params(self, silent=False):
        self.rot_slider.setValue(0)
        self.scale_slider.setValue(100)
        if self.original_img is not None:
            self.current_img = self.original_img.copy()
            self.update_display()
        if not silent:
            self.status_bar.showMessage("Параметры сброшены")

    def reset_all(self):
        self.reset_params()
        self.clear_points()
        self.status_bar.showMessage("Всё сброшено")

    def save_image(self):
        if self.current_img is None:
            QMessageBox.warning(self, "Ошибка", "Сначала откройте изображение")
            return

        # ИСПРАВЛЕНО: добавлен фильтр по умолчанию и проверка расширения
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not file_path:
            return

        # Если пользователь не указал расширение — добавляем его автоматически
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            if 'JPEG' in selected_filter:
                file_path += '.jpg'
            else:
                file_path += '.png'

        img_bgr = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, img_bgr)
        self.status_bar.showMessage(f"Сохранено: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())