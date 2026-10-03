# Contributing

Bug reports with the input image, the command you ran and the output are the most useful thing you can send.

To work on the code:

```
pip install -e ".[dev]"
pytest
```

Please add a test for any behaviour you change. Keep the base install light: new heavy dependencies belong behind an extra.
Ideas that would help: temporal smoothing for video, more styles, tests on macOS with a real camera.
