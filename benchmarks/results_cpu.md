python 3.10.6, Windows-10-10.0.26200-SP0, device: cpu

| case | first call (loads model) | median of warm calls |
| --- | --- | --- |
| built-in style, 256x256 | 0.06 s | 18 ms |
| built-in style, 512x512 | 0.10 s | 106 ms |
| built-in style, 800x800 | 0.23 s | 251 ms |
| built-in style, 1024x1024 | 0.63 s | 584 ms |
| style image, 384px network | 0.28 s | 103 ms |
| contact sheet (26 styles) | 0.75 s | 760 ms |
