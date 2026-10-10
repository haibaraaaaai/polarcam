# Camera update — awaiting lab validation

These changes were reviewed as source only. **No tests, GUI launch or camera execution
were performed**, as requested. Do not treat this commit as validated acquisition code.

## Recording layout and meaning

Launch with `Launch-Polarcam.cmd`. The data root is `data/`; choose hinges or bfm in
the shared header. A new app session writes to
`data/<experiment>/polarcam/YYYY-MM-DD_HHMMSS/`, with `frames/`, `spots/` and
`stationary/` subfolders. Session folder selection in any tab changes the common
destination. Existing imported files and folders are not renamed by the app.
Active background profiles still live under the data root's `backgrounds/`; select
or capture the appropriate profile for the current illumination and exposure.

- `pending`: stationary captures waiting for a human review.
- `good`: explicitly accepted by the operator, not automatically validated.
- `bad`: explicitly rejected by the operator; files are retained.
- `temp` and `attempt240nmsoundonbuffer`: old chosen/hard-coded destinations. Their
  names do not determine sample identity or interference state; new defaults no
  longer use them.
- Manual stationary review writes a separate `review.json` with decision and time.
  Raw recording metadata retains its acquisition-time unreviewed status.

There is one manual drive declaration (Unknown/On/Off). It describes the operator's
statement, not hardware feedback and not proof that cancellation works. The legacy
`sound.sound_on` field is derived from it; Unknown becomes JSON null. Shared
recording role and notes are captured at acquisition start for spot recordings as
well as frame stacks, and are stored in sidecars. Include sample ID, illumination,
laser power and paired-recording ID in the notes. Old metadata is untouched.

## Controls

- Default exposure is **0.6 ms**, default analogue gain **20**. Exposure and gain
  are shared across live, frame-stack, magnifier and spot capture controls. Apply
  settings to update an already running live controller; read back actual values
  before trusting the request.
- Magnifier ROI/settings can be applied explicitly. A running magnifier analysis
  is stopped and restarted so one trace does not mix different ROI definitions.
  Finish tuning/diagnostic trials before changing it.
- Click a rod in live view and use “Use magnifier centre for Spot examine”; a
  frame-stack detection is no longer required just to select a recording target.
- Spot preview shows the requested ROI before recording and the acquired crop
  during recording. Edit the ROI size/centre before the next recording. Camera ROI
  remains fixed during each recording. Sizes are even sensor pixels; odd requests
  round down. Invalid input must be corrected before acquisition.
- Add/remove buttons edit the loaded-file spot list and rerun the existing
  analysis pipeline so channel arrays and centres stay aligned. Added spots bypass
  automatic motion/ring/directionality filters. Manual edits last for the current
  app session and do not overwrite raw input. Analysis exports record the edits.
- The magnifier time plot can show mean intensity, X or Y anisotropy. X=(I0−I90)/
  (I0+I90), Y=(I45−I135)/(I45+I135), using ROI-origin parity and the displayed
  background-subtraction setting. These are diagnostics, not corrected angles.

## Stage 4: in-lab checks to perform

1. Confirm startup at 0.6 ms / gain 20 and inspect camera readback in saved JSON.
   Change exposure in one tab and verify the shared fields and actual capture.
2. Select hinges/bfm and a session folder; save a tiny stack and a short spot
   recording. Check paths, names, frame counts, FPS, background and ROI metadata.
3. Set drive Unknown, On and Off in turn. Confirm one consistent declaration in
   frame and spot sidecars, including sample/illumination/pair notes. The GUI must
   not imply that it switches the physical interference drive.
4. Move the live target, resize the ROI, start magnifier analysis and apply a new
   size. Check exposure stays matched; confirm X/Y change with the selected area.
5. In Spot examine, set ROI centre/size, record briefly, and compare the preview
   with the saved crop and sensor origin/parity. Stop before moving/resizing.
6. Load a short saved stack; add a spot to an empty area, remove a selected spot,
   and check map, traces and exported channel-array dimensions stay consistent.
7. Accept/reject a disposable stationary capture. Check files remain recoverable
   and `review.json` describes the manual action.
8. Check stopping, repeated capture, camera reconnect and quit. Record failures
   before taking a scientific dataset. Keep the preceding commit available for
   comparison; no history reset or data deletion is needed.

Large GUI restructuring, automatic biological quality classification and changing
the interference-drive amplitude are outside this preliminary change.
