python 3.10.6, Windows-10-10.0.26200-SP0, device: cuda

| case | first call (loads model) | median of warm calls |
| --- | --- | --- |
| built-in style, 256x256 | 2.23 s | 7 ms |
| built-in style, 512x512 | 0.04 s | 20 ms |
| built-in style, 800x800 | 0.06 s | 53 ms |
| built-in style, 1024x1024 | 0.12 s | 84 ms |
| style image, 384px network | 0.74 s | 102 ms |
| contact sheet (26 styles) | 0.59 s | 536 ms |
