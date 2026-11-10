# Запуск на видеокарте NVIDIA

Код сам выбирает устройство: Apple MPS → NVIDIA CUDA → CPU. На NVIDIA
ничего менять не нужно, достаточно поставить PyTorch с поддержкой CUDA.

Главное отличие от Mac: NVIDIA умеет float64. Поэтому на CUDA замер
добавляет GPU-варианты в двойной точности и сетку 1600 × 1600. Так GPU
и CPU сравниваются при одинаковой точности.

## 1. Что нужно

- Видеокарта NVIDIA и свежий драйвер. Проверка: `nvidia-smi`.
  В правом верхнем углу вывода указана «CUDA Version» — максимальная
  версия CUDA, которую поддерживает драйвер.
- Python 3.10–3.13.
- Отдельно ставить CUDA Toolkit не нужно: PyTorch приносит его с собой.

## 2. Установка

Скопируйте папку проекта на компьютер с видеокартой (без `.venv`,
`__pycache__` и файлов `*.pt`, `*.npy`).

**Windows (PowerShell):**

```powershell
cd cuda-learning
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Linux:**

```bash
cd cuda-learning
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`cu126` — сборка под CUDA 12.6, подходит драйверам с «CUDA Version» 12.6
и выше. Для старого драйвера (12.1–12.5) возьмите `cu121`/`cu124`, для
CUDA 13 — `cu130`. Точную команду для своей системы можно взять на
https://pytorch.org/get-started/locally/.

**Google Colab** (если своей видеокарты нет): Runtime → Change runtime
type → GPU (T4). Загрузите файлы проекта и выполните в ячейке
`!python laplace_gpu_benchmark.py`. PyTorch с CUDA там уже стоит.

## 3. Проверка, что GPU виден

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно быть `True` и имя видеокарты. Если `False` — установился PyTorch
без CUDA: `pip uninstall torch` и повторить установку с `--index-url`.

Затем все проверки (включая сравнение PCR с прогонкой на GPU):

```bash
python checks/run_checks.py
```

В конце должно быть `ALL CHECKS PASSED`.

## 4. Замер

```bash
python laplace_gpu_benchmark.py            # сетки 100, 200, 400, 800
python laplace_gpu_benchmark.py 100        # только 100 × 100 (задание диплома)
python laplace_gpu_benchmark.py 1600       # большая сетка
```

Первая строка вывода показывает, на чём идёт счёт, например
`GPU: NVIDIA GeForce RTX 4070 (cuda)`. Результаты сохраняются в
`results/laplace_gpu_benchmark_cuda.csv` (у Mac свой файл `_mps.csv`,
они не затирают друг друга). Файл в UTF-8 и открывается в Excel.

На CUDA появляются варианты:

| Вариант | Что сравнивает |
|---|---|
| `CG 5 точек GPU f64` | традиционный метод на GPU в двойной точности |
| `Коорд. GMRES+ADI, PCR GPU f64` | координация на GPU в двойной точности |
| `Коорд. GMRES+ADI, BLAS GPU f64` | то же с плотными треугольными матрицами |
| `… GPU` (без f64) | одинарная точность, как на Mac |

Сетка 1600 × 1600 занимает несколько минут: CG на CPU — около минуты,
координации на CPU нужно 4–6 ГБ оперативной памяти.

## 5. Что ожидать

- Игровые видеокарты (GeForce RTX) считают float64 в 32–64 раза медленнее
  float32. Для честного сравнения в двойной точности лучше серверная
  карта (A100, H100, V100) — например, в облаке или в университетском
  кластере. На GeForce float64-варианты всё равно стоит запустить:
  это и есть реальная картина для обычного компьютера.
- На сетке 100 × 100 GPU, скорее всего, снова не выиграет: задача слишком
  мала. Интересны сетки от 400 × 400.
- В столбце `gpu_mib` на CUDA — пиковый объём памяти, выделенный PyTorch
  (на Mac там был объём драйвера, поэтому числа напрямую не сравнимы).

## 6. Если что-то не так

- `CUDA out of memory` — уберите 1600 из запуска или закройте другие
  программы, использующие видеокарту.
- `torch.cuda.is_available()` = `False` при работающем `nvidia-smi` —
  стоит CPU-сборка PyTorch (см. раздел 3).
- В WSL на Windows нужен драйвер NVIDIA для Windows; внутри WSL ставить
  драйвер не надо.
