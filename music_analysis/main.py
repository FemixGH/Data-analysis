import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import soundfile as sf
import pyloudnorm as pyln
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial


warnings.filterwarnings('ignore')


class AudioAnalyzer:
    def __init__(self, input_path):
        self.input_path = input_path
        # Уникальное имя для каждого процесса
        self.wav_path = f"temp_audio_{os.getpid()}.wav"
        self.sr = 22050
        self.track_name = self._extract_track_name()
        self._convert_to_wav()
        self.y, self.sr = librosa.load(self.wav_path, sr=self.sr)

    def _extract_track_name(self):
        base = os.path.basename(self.input_path)
        return os.path.splitext(base)[0]

    def _convert_to_wav(self):
        if self.input_path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(self.input_path)
            audio.export(self.wav_path, format="wav")
        else:
            self.wav_path = self.input_path

    def analyze_all(self):
        results = {}
        chroma = self._get_chroma_features()
        results.update(self._analyze_harmonic_complexity(chroma))
        results.update(self._analyze_timbre_diversity())
        results.update(self._measure_loudness())
        return results

    def _get_chroma_features(self):
        return librosa.feature.chroma_stft(y=self.y, sr=self.sr)

    def _analyze_harmonic_complexity(self, chroma):
        threshold = np.mean(chroma)
        active_notes = chroma > threshold
        return {
            'harmonic_complexity': {
                "mean_active_notes": np.mean(np.sum(active_notes, axis=0)),
                "change_frequency": np.sum(np.abs(np.diff(active_notes.astype(int), axis=1))) / chroma.shape[1],
                "unique_chords": self._cluster_chords(chroma.T)
            }
        }

    def _cluster_chords(self, features):
        features = features[np.max(features, axis=1) > 0.1]
        if len(features) < 10:
            return 1

        best_score = -1
        optimal_clusters = 1

        for n in range(2, 9):
            kmeans = KMeans(n_clusters=n)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                optimal_clusters = n

        return optimal_clusters

    def _analyze_timbre_diversity(self):
        features = self._extract_timbre_features()
        X = features.T
        weights = np.ones(X.shape[1])
        return {
            'timbre_diversity': {
                "weighted_std": np.average(np.std(X, axis=0), weights=weights),
                "dynamic_range": np.mean(np.max(X, axis=0) - np.min(X, axis=0))
            }
        }

    def _extract_timbre_features(self):
        return np.vstack([
            librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13),
            librosa.feature.spectral_centroid(y=self.y, sr=self.sr),
            librosa.feature.spectral_contrast(y=self.y, sr=self.sr),
            librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr),
            librosa.feature.spectral_rolloff(y=self.y, sr=self.sr),
            librosa.feature.zero_crossing_rate(self.y)
        ])

    def _measure_loudness(self):
        data, rate = sf.read(self.wav_path)
        meter = pyln.Meter(rate)
        return {'loudness': meter.integrated_loudness(data)}


def process_single_song(row_tuple, output_csv):
    """Обработка одной песни с возвратом результатов"""
    try:
        idx, row = row_tuple

        # Создаем уникальное имя файла для избежания конфликтов
        filename = f"temp_{os.getpid()}_{idx}.mp3"
        results = {}

        # Формирование поискового запроса
        query = f"{row['title']} {row['performer']}"
        print(f"Обработка {idx + 1}: {row['title']} - {row['performer']}")

        # Поиск и скачивание трека
        search_url = 'https://rus.hitmotop.com/search'
        params = {'q': query}
        response = requests.get(search_url, params=params)

        if response.status_code != 200:
            print(f"Ошибка поиска: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        download_link = soup.find('a', class_='track__download-btn')

        if not download_link:
            print("Ссылка для скачивания не найдена")
            return None

        download_url = download_link['href']

        # Скачивание файла
        file_response = requests.get(download_url)
        if file_response.status_code != 200:
            print("Ошибка скачивания файла")
            return None

        with open(filename, "wb") as f:
            f.write(file_response.content)

        # Анализ аудио
        analyzer = AudioAnalyzer(filename)
        results = analyzer.analyze_all()


        os.remove(filename)
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")


        return (idx, {
            'loudness': results.get('loudness', np.nan),
            'mean_active_notes': results.get('harmonic_complexity', {}).get('mean_active_notes', np.nan),
            'change_frequency': results.get('harmonic_complexity', {}).get('change_frequency', np.nan),
            'unique_chords': results.get('harmonic_complexity', {}).get('unique_chords', np.nan),
            'weighted_std': results.get('timbre_diversity', {}).get('weighted_std', np.nan),
            'dynamic_range': results.get('timbre_diversity', {}).get('dynamic_range', np.nan)
        })

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return None

def process_songs(csv_path, output_csv):

    df = pd.read_csv(csv_path)
    # df = df.head(10)
    new_columns = ['loudness', 'mean_active_notes', 'change_frequency',
                   'unique_chords', 'weighted_std', 'dynamic_range']

    for col in new_columns:
        if col not in df.columns:
            df[col] = np.nan

    # пул процессов
    num_workers = max(1, cpu_count() - 1)


    if __name__ == '__main__':
        with Pool(processes=num_workers) as pool:
            # Запускаем обработку в параллель
            results = pool.imap_unordered(
                partial(process_single_song, output_csv=output_csv),
                df.iterrows(),
                chunksize=2
            )

            # Собираем результаты
            for result in results:
                if result:
                    idx, data = result
                    for col in new_columns:
                        df.at[idx, col] = data.get(col, np.nan)

                    # Промежуточное сохранение каждые 10 записей
                    if idx % 10 == 0:
                        df.to_csv(output_csv, index=False)

        # Финальное сохранение
        df.to_csv(output_csv, index=False)
        print(f"Результаты сохранены в {output_csv}")


if __name__ == "__main__":
    process_songs("top_100_songs_by_year.csv", "parallel_results.csv")