OCR this Vietnamese document page into Markdown. Output the transcription only: no preamble, no commentary.

**Reading order.** Follow the page's natural reading order; never interleave columns.

**Blank pages.** If the page carries no text, return nothing at all.

**Figures.** Skip all graphical content: maps and diagrams, emblems and logos, photographs, fingerprints, QR codes, circle seals, north arrows and decorative borders.

**Two placeholders, and no others.**
- `[Chữ ký]` — where a handwritten signature appears (alone or over a seal). Put the printed name on the next line if one is printed.
- `[Mã vạch]` — where a barcode appears. Put its printed digits on the next line.

**Headings.** `#` title, `##` sections, one more `#` per level down. Rank by type size first, numbering depth second; never skip a level.

**Emphasis.** Plain text — no bold, italic or underline. One exception: `~~struck out~~` for text crossed out on the page, e.g. a voided entry.

**Stamps and overlays.** Text inside a rectangular stamp or a box stamped over the page — `BẢN SAO`, `SAO Y BẢN CHÍNH`, a certification block, an incoming-mail stamp, a handwritten note in a box — goes in a fenced block. Nothing else is fenced: never fence the page's own body text, a letterhead column, or a heading.

**Tables.**
- Default to a Markdown table.
- Merged cells are fine as long as they flatten cleanly: repeat a spanned value into each cell it covers, and join a stacked header into one header line. Apply the joined prefix to every column it covers, not just the first.
- When one column's entries sit between the rows of another — edge lengths listed between vertex labels — keep the offset: each gets its own row with the other cell left empty. Do not pair them up.
- Fall back to HTML `<table>` only when flattening would lose structure.
- Use `<br>` for line breaks inside a cell.
