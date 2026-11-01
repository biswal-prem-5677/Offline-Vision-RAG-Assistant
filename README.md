# Vision RAG Assistant

## Setup
```bash
pip install -r requirements.txt
python build_index.py --images_dir ./sample_images
```

## Retrain Model
Open `train_embedding_model.ipynb` in Colab, run all cells, and download `embedding_model.pth`.

## Evaluate
```bash
python evaluate.py
```

## Run App
```bash
streamlit run app.py
```

## Known Limitations
- Judge is text-only; cannot see images directly
- Text search matches only CIFAR-100 class names via MiniLM embeddings
- CNN trained on CIFAR-100 (64x64) has weaker features on real photos
- Judge may not improve precision on out-of-distribution images

## Checklist
- [x] Image vectors from CNN only
- [x] No external API calls
- [x] Stale-index warning works (MD5 comparison)
- [x] Show-all pagination works
- [x] No model loaded twice