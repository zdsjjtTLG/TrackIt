# 发布 GoTrackIt 到 PyPI

本项目在 `main` 分支手动构建、手动上传 Python 包。

## 发布前检查

1. 确认当前分支为 `main`，且要发布的改动已提交。
2. 在 `pyproject.toml` 中递增 `project.version`，并同步更新 `src/gotrackit/__init__.py` 中的 `__version__`（如有）。
3. 每个版本号在 TestPyPI 和 PyPI 都只能上传一次；若已上传过，必须再次递增版本号。

## 构建

本项目的虚拟环境可能没有安装 `pip` 或 `build`。推荐使用 `uv`：

```bash
uv build
```

构建产物位于 `dist/`，应包括同一版本的源码包和 wheel，例如：

```text
dist/gotrackit-0.3.26.tar.gz
dist/gotrackit-0.3.26-py3-none-any.whl
```

如需使用标准 Python 构建命令，先安装 `build`：

```bash
uv pip install build
python -m build
```

## 上传 TestPyPI

将 TestPyPI token 放到环境变量中（不要把 token 直接写入命令或提交到仓库）：

```bash
export TEST_PYPI_TOKEN='pypi-...'
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --token "$TEST_PYPI_TOKEN" \
  dist/gotrackit-<版本号>*
```

在干净环境中验证安装；`--extra-index-url` 允许依赖从正式 PyPI 获取：

```bash
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  gotrackit==<版本号>
```

## 上传正式 PyPI

确认 TestPyPI 安装和基本功能正常后，**直接上传刚才构建在 `dist/` 中的同一批产物，无需重新执行 `uv build`**：

```bash
export PYPI_TOKEN='pypi-...'
uv publish \
  --token "$PYPI_TOKEN" \
  dist/gotrackit-<版本号>*
```

例如当前版本为 `0.3.26` 时：

```bash
uv publish --token "$PYPI_TOKEN" dist/gotrackit-0.3.26*
```

该命令会同时上传源码包（`.tar.gz`）和 wheel（`.whl`）。

上传后确认：

```bash
pip install --upgrade -i https://pypi.org/simple/ gotrackit==<版本号>
```

## 注意事项

- 发布前可删除旧的构建产物，或明确指定本次版本号的两个文件，避免误传旧版本。
- PyPI token 如出现在终端历史、日志或聊天记录中，应立即在 PyPI 撤销并重新生成。
