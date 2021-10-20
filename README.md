# Demixing-Channel

## Preprocess step one data

```python
python preprocess.py
```

Return pickle file:

```
{
    train: List[(wav: np.array, speaker: str)]
    test: List[(wav: np.array, speaker: str)]
}
```