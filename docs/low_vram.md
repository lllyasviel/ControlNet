# Enable Low VRAM Mode

If you are using 8GB GPU card (or if you want larger batch size), please open "config.py", and then set

```python
save_memory = True
```

This feature is still being tested - not all graphics cards are guaranteed to succeed.

But it should be neat as I can diffuse at a batch size of 12 now.

(prompt "man")

![p](../github_page/ram12.jpg)
