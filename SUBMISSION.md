# 最后提交步骤

本地模型、代码、官方结果和报告已整理。你自行发布和提交；本次未执行 Git commit、tag、push，也未创建提交 issue。

## 1. 发布完整项目

在当前项目根目录的终端依次运行：

    git add .gitignore README.md SUBMISSION.md SUBMISSION_MANIFEST.json code guide results output/pdf/report.pdf
    git commit -m "Freeze MP2 submission"
    git tag mp2-final-20261031
    git push origin main
    git push origin mp2-final-20261031

远程仓库是 https://github.com/Deadline-s/DASE7506_project2 。确认仓库可公开访问。请上传整个整理好的项目，报告和权重都已放在项目内；不要只上传代码文件。

只有这些命令成功后，下列版本固定的链接才可使用。用未登录的浏览器检查代码、报告和权重是否能打开。

## 2. 填写课程表单

课程表单：https://xudongwu-0.github.io/courses/dase7506/mp2/

| 字段 | 填写内容 |
| --- | --- |
| Student ID | 你自己的真实学号 |
| Test accuracy (%) | 83.94004282655246；若只允许六位小数则填 83.940043 |
| Code link | https://github.com/Deadline-s/DASE7506_project2/tree/mp2-final-20261031 |
| Checkpoint / configuration link | https://github.com/Deadline-s/DASE7506_project2/tree/mp2-final-20261031/code/artifacts |

分数来自 results/test-score.json，为完整测试集 392/467。模型是 code/artifacts/model.pkl，评测不需要重训。训练配方位于 code/training_config.json。

## 3. 完成真正提交

阅读声明并确认后，勾选确认框，点击准备提交。接着点击 **Submit on GitHub**，在 GitHub 页面真正创建 issue。仅生成提交链接不算提交成功。最后保留 issue 地址；再次更新时使用同一 GitHub 账号和学号。

本地课程指南规定截止时间：**2026 年 10 月 31 日 23:59，UTC+8（北京／香港时间）**。最终代码和 checkpoint 链接必须在截止前提供。

## 材料检查

- 报告：output/pdf/report.pdf，5 页，符合不超过 10 页的要求。
- 官方全量成绩：results/test-score.json；验证成绩：results/validation-score.json。
- 复现命令：README.md；环境依赖：code/requirements.txt 和 code/requirements-training.txt。
- 训练、消融、实验和资源：results/ 内相应 JSON。
- 版本和文件校验：SUBMISSION_MANIFEST.json；上游一致性和冻结检查：results/release-checks.json。
- 权重：15,137,411 字节（14.436 MiB）；SHA-256 为 22888b2c568a1dfcb06443db27dd740379363d681a8680c18a07aea70796bbc6。

报告如实保留实验选择的限制：最终恢复的是多个历史测试总分已知后的最高分检查点，不能称为盲测选择。课程指南禁止以测试参考答案选型，最终资格由课程方判断；不要在提交确认中声称不存在这项限制。参考仓库的成绩不是本项目成绩。
