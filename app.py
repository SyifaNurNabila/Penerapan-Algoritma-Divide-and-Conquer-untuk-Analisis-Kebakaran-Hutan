import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

@dataclass
class FireZone:
    """Representasi area kebakaran yang terdeteksi"""
    x: int  # Posisi x
    y: int  # Posisi y
    width: int  # Lebar area
    height: int  # Tinggi area
    intensity: float  # Intensitas api (0-1)
    risk_level: str  # Level risiko (low/medium/high/critical)
    confidence: float  # Tingkat kepercayaan deteksi (0-1)
    area: float  # Luas area

class ForestFireDetector:
    def __init__(self, min_block_size: int = 32):
        self.min_block_size = min_block_size
        
        # Parameter untuk deteksi api
        self.fire_ranges = [
            # Format: ((Hue min, max), (Saturation min, max), (Value min, max))
            ((0, 50), (150, 255), (200, 255)),  # Api terang
            ((0, 40), (130, 255), (180, 255)),  # Api sedang
            ((0, 60), (100, 255), (150, 255))   # Api redup
        ]
        
        # Parameter untuk deteksi asap
        self.smoke_ranges = [
            ((0, 180), (0, 30), (180, 255)),    # Asap putih
            ((0, 180), (0, 50), (150, 220))     # Asap abu-abu
        ]

    def load_image(self, path: str) -> np.ndarray:
        """Memuat dan melakukan preprocessing awal pada gambar"""
        # Baca gambar
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Tidak dapat memuat gambar: {path}")
            
        # Preprocessing
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
        return img

    def detect_fire_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mendeteksi area api dan asap dalam gambar"""
        # Konversi ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Deteksi api
        fire_mask = np.zeros_like(hsv[:,:,0])
        for h_range, s_range, v_range in self.fire_ranges:
            lower = np.array([h_range[0], s_range[0], v_range[0]])
            upper = np.array([h_range[1], s_range[1], v_range[1]])
            mask = cv2.inRange(hsv, lower, upper)
            fire_mask = cv2.bitwise_or(fire_mask, mask)
            
        # Deteksi asap
        smoke_mask = np.zeros_like(hsv[:,:,0])
        for h_range, s_range, v_range in self.smoke_ranges:
            lower = np.array([h_range[0], s_range[0], v_range[0]])
            upper = np.array([h_range[1], s_range[1], v_range[1]])
            mask = cv2.inRange(hsv, lower, upper)
            smoke_mask = cv2.bitwise_or(smoke_mask, mask)
            
        # Gabungkan hasil deteksi
        combined_mask = cv2.addWeighted(fire_mask, 0.8, smoke_mask, 0.2, 0)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = gaussian_filter(combined_mask, sigma=1)
        
        return combined_mask, hsv

    def analyze_block(self, block: np.ndarray, hsv_block: np.ndarray) -> float:
        """Menganalisis intensitas api dalam suatu block"""
        # Hitung statistik blok
        mean_v = np.mean(hsv_block[:,:,2])
        std_v = np.std(hsv_block[:,:,2])
        mean_s = np.mean(hsv_block[:,:,1])
        
        # Hitung skor berdasarkan karakteristik api
        intensity = (mean_v / 255.0) * 0.4 + (mean_s / 255.0) * 0.4 + (std_v / 128.0) * 0.2
        
        return np.clip(intensity, 0, 1)

    def divide_and_analyze(self, mask: np.ndarray, hsv: np.ndarray) -> List[FireZone]:
        """Implementasi divide and conquer untuk analisis gambar"""
        height, width = mask.shape[:2]
        zones = []
        
        def recursive_analyze(x: int, y: int, w: int, h: int):
            if w <= self.min_block_size or h <= self.min_block_size:
                return
                
            block = mask[y:y+h, x:x+w]
            hsv_block = hsv[y:y+h, x:x+w]
            
            # Analisis blok
            if np.mean(block) > 20:  # Minimal aktivitas dalam blok
                intensity = self.analyze_block(block, hsv_block)
                
                if intensity > 0.3:  # Threshold minimal untuk deteksi
                    # Hitung confidence score
                    confidence = self.calculate_confidence(block, hsv_block, intensity)
                    
                    if confidence > 0.4:  # Filter berdasarkan confidence
                        # Tentukan risk level
                        risk_level = self.assess_risk_level(intensity, confidence, w * h)
                        
                        zones.append(FireZone(
                            x=x, y=y,
                            width=w, height=h,
                            intensity=intensity,
                            confidence=confidence,
                            risk_level=risk_level,
                            area=w * h
                        ))
                        return
            
            # Recursive division
            half_w = w // 2
            half_h = h // 2
            
            recursive_analyze(x, y, half_w, half_h)  # Top-left
            recursive_analyze(x + half_w, y, half_w, half_h)  # Top-right
            recursive_analyze(x, y + half_h, half_w, half_h)  # Bottom-left
            recursive_analyze(x + half_w, y + half_h, half_w, half_h)  # Bottom-right
            
        recursive_analyze(0, 0, width, height)
        return self.merge_zones(zones)

    def calculate_confidence(self, block: np.ndarray, hsv_block: np.ndarray, 
                           intensity: float) -> float:
        """Menghitung tingkat kepercayaan deteksi"""
        # Analisis warna
        color_score = np.mean(hsv_block[:,:,1]) / 255.0  # Saturation
        
        # Analisis tekstur
        gradient_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
        texture_score = np.mean(np.sqrt(gradient_x**2 + gradient_y**2)) / 255.0
        
        # Kombinasikan skor
        confidence = (
            0.4 * intensity +
            0.3 * color_score +
            0.3 * texture_score
        )
        
        return np.clip(confidence, 0, 1)

    def assess_risk_level(self, intensity: float, confidence: float, area: int) -> str:
        """Menentukan tingkat risiko kebakaran"""
        # Normalisasi area
        area_score = min(1.0, area / (256 * 256))
        
        # Hitung total skor
        risk_score = (
            0.4 * intensity +
            0.3 * confidence +
            0.3 * area_score
        )
        
        # Tentukan level risiko
        if risk_score > 0.8:
            return 'critical'
        elif risk_score > 0.6:
            return 'high'
        elif risk_score > 0.3:
            return 'medium'
        else:
            return 'low'

    def merge_zones(self, zones: List[FireZone], distance_threshold: int = 30) -> List[FireZone]:
        """Menggabungkan zona yang overlap atau berdekatan"""
        if not zones:
            return []
            
        merged = []
        used = set()
        
        for i, zone1 in enumerate(zones):
            if i in used:
                continue
                
            current_group = [zone1]
            used.add(i)
            
            for j, zone2 in enumerate(zones):
                if j in used:
                    continue
                    
                # Cek jarak antar zona
                dist = np.sqrt(
                    (zone1.x - zone2.x)**2 +
                    (zone1.y - zone2.y)**2
                )
                
                if dist < distance_threshold:
                    current_group.append(zone2)
                    used.add(j)
            
            # Gabungkan grup
            if current_group:
                merged.append(self._combine_zones(current_group))
                
        return merged

    def _combine_zones(self, zones: List[FireZone]) -> FireZone:
        """Menggabungkan beberapa zona menjadi satu"""
        # Hitung bounding box
        min_x = min(z.x for z in zones)
        min_y = min(z.y for z in zones)
        max_x = max(z.x + z.width for z in zones)
        max_y = max(z.y + z.height for z in zones)
        
        # Hitung rata-rata properti
        avg_intensity = np.mean([z.intensity for z in zones])
        avg_confidence = np.mean([z.confidence for z in zones])
        total_area = sum(z.area for z in zones)
        
        # Ambil risk level tertinggi
        risk_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        max_risk = max(zones, key=lambda z: risk_levels[z.risk_level]).risk_level
        
        return FireZone(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            intensity=avg_intensity,
            confidence=avg_confidence,
            risk_level=max_risk,
            area=total_area
        )

    def detect_fire(self, image_path: str) -> List[FireZone]:
        """Method utama untuk deteksi kebakaran"""
        # Load dan preprocessing
        image = self.load_image(image_path)
        
        # Deteksi area api dan asap
        fire_mask, hsv = self.detect_fire_regions(image)
        
        # Analisis dengan divide and conquer
        return self.divide_and_analyze(fire_mask, hsv)


    def visualize_results(self, image_path: str, fire_zones: List[FireZone]) -> None:
        """Visualisasi hasil deteksi"""
        image = self.load_image(image_path)
        
        # Warna untuk setiap risk level
        colors = {
            'low': (255, 255, 255),   # Putih untuk asap
            'medium': (0, 255, 255),  # Kuning
            'high': (0, 128, 255),    # Oranye
            'critical': (0, 0, 255)   # Merah
        }
        
        # Buat overlay
        overlay = image.copy()
        
        total_area = 0
        avg_confidence = 0
        max_intensity = 0
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for zone in fire_zones:
            color = colors[zone.risk_level]
            
            # Gambar box dengan alpha berdasarkan confidence
            cv2.rectangle(
                overlay,
                (zone.x, zone.y),
                (zone.x + zone.width, zone.y + zone.height),
                color,
                2
            )
            
            # Tambah label
            label = f"{zone.risk_level} ({zone.confidence:.2f})"
            cv2.putText(
                overlay,
                label,
                (zone.x, zone.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
            # Update statistics
            total_area += zone.area
            avg_confidence += zone.confidence
            max_intensity = max(max_intensity, zone.intensity)
            risk_counts[zone.risk_level] += 1
        
        avg_confidence /= len(fire_zones)
        
        # Blend overlay dengan gambar asli
        result = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Tampilkan hasil
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Hasil Deteksi Kebakaran Hutan')
        
        # Tambahkan teks statistik ke plot
        stats_text = (
            f"Total area terdeteksi: {total_area:.0f} pixel²\n"
            f"Rata-rata confidence: {avg_confidence:.2f}\n"
            f"Intensitas maksimum: {max_intensity:.2f}\n"
            "\nDistribusi tingkat risiko:\n" +
            "\n".join([f"- {risk_level}: {count} zona" for risk_level, count in sorted(risk_counts.items())])
        )
        plt.gcf().text(0.01, 0.75, stats_text, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
        
        plt.show()


class FireDetectionApp:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Forest Fire Detection")
        
        self.detector = ForestFireDetector(min_block_size=32)
        
        self.label = tk.Label(root, text="Pilih gambar untuk mendeteksi kebakaran hutan:")
        self.label.pack(pady=10)
        
        self.select_button = tk.Button(root, text="Pilih Gambar", command=self.select_image)
        self.select_button.pack(pady=5)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(initialdir="data/", title="Pilih Gambar",
                                               filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
        if file_path:
            self.detect_fire(file_path)
    
    def detect_fire(self, image_path):
        try:
            fire_zones = self.detector.detect_fire(image_path)
            
            if not fire_zones:
                messagebox.showinfo("Hasil Deteksi", "Tidak ada kebakaran terdeteksi.")
                os.remove(image_path)
                self.result_label.config(text="Tidak ada kebakaran terdeteksi. Gambar dihapus.")
                self.image_label.config(image='')
                return
            
            self.result_label.config(text="Kebakaran terdeteksi.")
            
            # Tampilkan gambar
            img = Image.open(image_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
            
            # Visualisasi hasil
            self.detector.visualize_results(image_path, fire_zones)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()