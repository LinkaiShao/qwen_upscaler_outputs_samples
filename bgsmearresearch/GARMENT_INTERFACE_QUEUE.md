
## queue start Thu Jul  9 11:20:10 PM PDT 2026
[23:20:10] TRAIN BGBAND52_0709_232010 — Run1: late-block residual xattn @ 50,55,59
  NO FINAL — training failed, skipping
        torch.save({k: v.cpu() for k, v in state._LATE_XATTN.state_dict().items()}, f"{out}/late_xattn.pt")
                                                                                       ^^^
    NameError: name 'out' is not defined. Did you mean: 'oct'?
[02:34:22] TRAIN BGBAND53_0710_023422 — Run6: 4 late sites + strong outside/boundary penalty
  NO FINAL — training failed, skipping
        torch.save({k: v.cpu() for k, v in state._LATE_XATTN.state_dict().items()}, f"{out}/late_xattn.pt")
                                                                                       ^^^
    NameError: name 'out' is not defined. Did you mean: 'oct'?
[05:48:34] TRAIN BGBAND54_0710_054834 — Run1-strong: gate forced to 1.0 (full branch authority)
  NO FINAL — training failed, skipping
        torch.save({k: v.cpu() for k, v in state._LATE_XATTN.state_dict().items()}, f"{out}/late_xattn.pt")
                                                                                       ^^^
    NameError: name 'out' is not defined. Did you mean: 'oct'?
[09:02:46] TRAIN BGBAND52_0710_090246 — Run1: late-block residual xattn @ 50,55,59
