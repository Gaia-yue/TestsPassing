## upload your embeddings model

```python
from modelscope.hub.snapshot_download import snapshot_download
model_name = ""
model_dir = snapshot_download(model_name=model_name, cache_dir='embedding_model')
```
