## Agent skills

### Issue tracker

Issues and specs live as GitHub issues, driven with the `gh` CLI. See `docs/agents/issue-tracker.md`.

**Issue 是写给一个没在电脑前看着你调试的人读的。** 用他不翻代码就能读懂的话写：

- **标题**一句话说清「谁、什么情况下、错在哪」。函数名、字段名、行号、百分比留给正文。
- **正文**开头几行先说现象——期望发生什么、实际发生什么——再说原因。读完这几行，读者要能复述出「什么坏了」，而不是「哪个函数坏了」。
- **术语**只用 `CONTEXT.md` 里收录的。其余的首次出现就一句话解释；解释不动就换成日常说法。
- **复现**给一条能跑的命令：现在跑出什么、本该跑出什么。
- **克制**加粗、表格、破折号。通篇加粗等于没有加粗；三五行说得清的事，别拆成五段加一张表。

反例（issue #73 的原标题）：

> mbt screen 用未复权价选股，而 backtest 用后复权——同一个 Screen 在两条路上给出不同答案

照这个标准写：

> 每日选股的名单和回测对不上：同一天，选股一条用未复权价、回测一条用后复权价

同一把尺子也管 issue 评论和 PR 描述——读者是同一个人。旧 issue 不必回头改。

### Triage labels

Five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the root plus `docs/adr/`. See `docs/agents/domain.md`.
