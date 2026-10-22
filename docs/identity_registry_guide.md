# 演员身份与剧目角色表

人脸匹配先确定演员身份，再用当前剧目的已确认演员表关联角色。同一演员的多个锚定目录共用一个 `identity_id`，计算最佳身份与次佳身份分差时只算一个候选。只有当前剧目存在唯一的已确认角色时，归档名称才写入“演员（饰角色）”。

## 在工作台使用

在“整理与导出”的“演员与角色表（可选）”填写服务端可访问的 JSON 文件路径。该设置随项目保存在浏览器中，同时用于整理和“演员与角色”中的专辑生成。填写路径只读取该表，不修改正式图库。

留空时读取人脸库根目录下的 `identity_registry.json`。默认文件不存在时兼容旧图库：从目录名提取演员名，身份来源标为 `gallery_name_unverified`，角色不从锚定目录继承。显式填写的文件不存在、不是普通文件或格式非法时，操作会报错。

“限定演员”支持图库目录名、目录路径、登记演员名的模糊匹配，以及精确的 `identity_id`；多人可用逗号或顿号分隔。同名演员需要指定身份 ID 才能单独选择。筛选只限制生成哪些专辑，其他演员仍参与识别竞争。

## 文件格式

以下为格式示例。实际使用时填写已核对的演员、目录和来源；`gallery_dirs` 是人脸库中的直接子目录名。

```json
{
  "schema_version": 1,
  "plays": [
    {"play_id": "play-a", "title": "京剧《示例甲》", "aliases": ["示例甲"]},
    {"play_id": "play-b", "title": "话剧《示例乙》"}
  ],
  "identities": [
    {
      "identity_id": "actor-001",
      "actor_name": "张演员",
      "verified": true,
      "gallery_dirs": ["张演员（饰旧角）", "张某（饰另角）"],
      "roles": [
        {
          "play_id": "play-a",
          "role_id": "role-a",
          "role_name": "甲角",
          "verified": true,
          "source_ref": "已核对演员表：示例甲，第1页"
        },
        {
          "play_id": "play-b",
          "role_id": "role-b",
          "role_name": "乙角",
          "verified": true,
          "source_ref": "已核对演员表：示例乙，第2页"
        }
      ]
    }
  ]
}
```

| 字段 | 规则 |
| --- | --- |
| `schema_version` | 当前为 `1` |
| `plays` / `identities` | 必须为数组；可为空 |
| `play_id` / `identity_id` / `role_id` | 非空，1–100 个英文字母、数字、点、下划线或连字符；剧目/身份 ID 各自唯一，角色 ID 在剧目内解释 |
| `title` / `aliases` | 剧目名称与可选别名；同一名称、别名或剧目 ID 不能关联两个剧目 |
| `actor_name` | 展示姓名；同名的不同演员使用不同身份 ID；演员改名时保留原身份 ID |
| `gallery_dirs` | 必填数组；只能写直接子目录名，不含路径分隔符、`.` 或 `..`；一个目录不能分配给多个身份 |
| 身份 `verified` | 必须为布尔值；只有 `true` 的身份绑定生效，未确认的绑定降级到旧目录名规则 |
| `roles` | 可选数组；每个角色须引用已存在的剧目，并填写 `role_id`、`role_name`、布尔 `verified` 和非空 `source_ref` |
| 角色 `verified` / `source_ref` | 仅使用已确认角色；`source_ref` 保存人工演员表、文档页码或其他实际核对依据 |

一位演员在不同剧目可以有不同角色。在同一剧目有多个已确认角色时，现有契约不能按单张照片区分具体角色，返回 `ambiguous_role`，名称只写演员。剧目优先从照片文件名读取，缺失时从目录名读取，再用表中的剧目 ID、名称或别名匹配；剧目未知或没有已确认角色时同样只写演员。

同一展示名称对应多个登记身份时，输出追加稳定的 `[identity_id]`，例如 `张演员 [actor-001]`。即使这些演员分别出现在不同照片或剧目中，也能保持专辑分离；不会等到两人同框才临时区分。唯一的演员/角色展示名称保留原来的简洁形式。

## API 参数

| 接口 | 演员表 | 相似度阈值 | 身份分差 |
| --- | --- | --- | --- |
| `POST /api/organize/tasks`，`kind="organize"` | `params.identity_registry_file` | `params.face_threshold` | `params.face_margin` |
| `POST /api/organize/run`，Form | `identity_registry_file` | `face_threshold` | `face_margin` |
| `POST /api/organize/actor_view`，Form | `identity_registry_file` | `threshold` | `margin` |

整理默认相似度阈值来自 `KNOWN_FACE_THRESHOLD`（默认 `0.55`，可用 `TRANS_KNOWN_FACE_THRESHOLD` 配置），专辑 API 的 `threshold` 默认 `0.55`；身份分差默认 `FACE_AMBIGUOUS_MARGIN=0.05`。阈值允许 `[0, 1]`，分差允许 `[0, 2]`，拒绝非有限数值。阈值和分差均满足才确认身份；只有一个有效身份候选时不计算第二名分差。这些值是匹配规则参数，不是经过正式校准的正确概率。

后台整理请求示例（JSON；路径需替换为服务端实际目录）：

```json
{
  "kind": "organize",
  "idempotency_key": "project-a-organize-001",
  "params": {
    "input_dir": "/photos/project-a",
    "output_dir": "/results/project-a-run-001",
    "db_root": "/galleries/actors",
    "identity_registry_file": "/projects/project-a/cast.json",
    "face_threshold": 0.55,
    "face_margin": 0.05,
    "classify_role": false,
    "keep_per_bucket": 2
  }
}
```

使用 `GET /api/organize/tasks/{task_id}` 查询结果。专辑接口可增加 `person=actor-001`，只生成指定身份的专辑；`copy_images=false` 时仍保存 `actor_view_report.json`，便于检查命中结果。

## 报告与恢复

- 整理的 `organize_report.json` 中，逐图 `recognized` 保留展示名称→分数；`identities` 增加稳定身份 ID、演员名、剧目/角色 ID、身份来源、角色依据、锚定目录及展示名称。
- `face_decisions` 按检测到的人脸下标记录每张脸的决定、分数及拒识原因，例如 `below_threshold`、`ambiguous_identity`、`no_valid_anchor`、`alignment_failed` 或 `empty_gallery`。一张脸至多匹配一个身份，同图同身份合并为最高分。
- 专辑报告逐图提供 `hits`、`identities`、`face_decisions`；报告及 API 摘要包含 `margin` 和 `identity_registry_sha256`。API 中的 `persons` 为各专辑的照片计数。
- `role_status` 为 `matched`、`unknown_play`、`ambiguous_role` 或 `no_verified_role`。未关联角色时，`role_id`、`role_name`、`role_source` 均为空。
- 整理任务绑定角色表完整内容摘要，并在处理和导出时复查；同大小、同修改时间的改写仍会被发现。改表后继续旧任务会报 `identity_registry_content_changed`，需要使用原版本继续或创建新任务。仅修改 JSON 排版也会改变摘要。

## 已确认文件名的导入

`tools/import_verified_photo_labels.py` 从 `data/ncpa_test/*/【整理完成】*` 的照片文件名提取已确认的图级幕次与人物/角色标签，核对 U0 清单内容摘要后生成独立产物。它不对文件名自行作真实性确认，适用范围应与实际人工确认一致。

```bash
/home/huachenghao/.conda/envs/face_scrfd_arcface/bin/python tools/import_verified_photo_labels.py \
  --root data/ncpa_test \
  --inventory outputs/acceptance/core_upgrade/first_pass/manifest.jsonl \
  --gallery data/face_database \
  --confirmation-source user_confirmed_organized_filenames_20261022 \
  --role-aliases outputs/acceptance/core_upgrade/verified_labels_20261022/confirmed_role_aliases.json \
  --output outputs/acceptance/core_upgrade/verified_labels_new_run
```

输出目录须为空且位于原图之外。产物包括 `verified_photo_labels.json`、`identity_registry.json` 和 `annotation_status.json`；原始角色名、已确认别名来源、多角色冲突及缺失标注均保留。导入身份按剧目和姓名生成，跨剧目同名不自动合并；只有姓名无歧义时绑定已有图库目录。

图级人物列表不等于框级人脸身份或完整人物清单。当前数据还不能用于已知/未知人脸误报率、连拍选优或时间边界的正式验收。实际标注版本、剩余冲突、质量评测结果和后续计划见 [工作状态](../WORK_STATE.md) 与 [核心升级进度](core_upgrade_progress.md)。
