# Context crossover for the Mojo model comparison

Goal: separate context-associated variation from a Mojo replacement effect. Existing blocking CUDA controls favor context0 by different amounts, so assigning Mojo only to context1 confounds interpretation. No whole-model gain is assumed.

Expose an explicit candidate context0|1, default1. Worker and controller report/check it in handshake and each trial. Each process still loads one model and two independent contexts. Assignment stays fixed for a process: do not reuse a captured CUDA graph as Mojo or vice versa. CUDA controls remain CUDA on both contexts regardless of assignment metadata.

The summarizer retains raw context ratios and adds explicitly directed CUDA-latency/Mojo-latency ratios for candidate runs; it emits no such ratio for controls. Wrong assignment metadata or trial workload must fail closed. Keep historical evidence readable with its recorded/default assignment1.

Protocol planned before outcomes: one CUDA control, four candidate attempts with Mojo contexts1,0,0,1, then one CUDA control. All use verified blocking waits, two priming tokens,16 timed tokens, one within-process ABBA cycle,20% CPU, same artifacts and unchanged cold/absolute thermal guards. Desktop quiet windows retain independent resume watchdogs. Stop the batch on a failed safety/cold gate, retain failures and do not select winning samples. Require zero timed replacement/capture counts for the intended primed comparison and matched fixed-prefix top tokens. These are exploratory fixed-token measurements, not sufficient alone to qualify full sampled output or PDFs.

Alternatives: subtracting a previous control's bias assumes it is stable; simply collecting more context1 trials preserves the confound. Crossover is the smaller change that directly tests it. No kernel, model or host defaults change. The next decision is whether any apparent gain follows Mojo, the context, or neither.
