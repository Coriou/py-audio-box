# Billing reconciliation — 20260825-165347

Generated: 2026-08-26T07:10:48.169760+00:00
Billing CSV: `/Users/ben/Projects/py-audio-box/work/sfx-bench/20260825-165347/fal-usage-20260825-v2.csv` (8 data rows)

> **aggregated billing export** — request-level join impossible; per-endpoint totals.

## Column mapping

| field | header | status |
|---|---|---|
| request_id | — | not detected |
| model | app_id | detected |
| cost | gross_amount | detected |
| currency | — | not detected |

Headers found: `timestamp`, `app_id`, `amount`, `quantity`, `unit`, `auth_method`, `unit_price`, `auth_method_label`, `gross_amount`, `net_amount`, `contracted_unit_price`, `list_unit_price`, `discount_percent`, `is_surge`, `price_category`

## Billing structure

| endpoint (app id) | quantity | unit | unit price | billed |
|---|---:|---|---:|---:|
| cassetteai/sound-effects-generator | 5 | generations | $0.01 | $0.0500 |
| cassetteai/video-sound-effects-generator | 5 | minutes | $0.2 | $1.0000 |
| fal-ai/controlfoley | 72 | seconds | $0.002 | $0.1440 |
| fal-ai/hunyuan-video-foley | 4.765273 | 10 seconds | $0.1 | $0.4765 |
| fal-ai/kling-video/video-to-audio | 5 | videos | $0.035 | $0.1750 |
| mirelo-ai/sfx1.6/video-to-video | 51 | seconds | $0.01 | $0.5100 |
| sonilo/v1.1/video-to-sound-effects | 50 | seconds | $0.009 | $0.4500 |

## Per-model billed vs estimated

| model | billed | estimated | delta | delta % |
|---|---:|---:|---:|---:|
| cassette-t2s | $0.0500 | $0.0500 | +0.0000 USD | +0.00% |
| cassette-v2s | $1.0000 | $0.1588 | +0.8412 USD | +529.56% |
| controlfoley | $0.1440 | $0.1141 | +0.0299 USD | +26.20% |
| hunyuan | $0.4765 | $0.4765 | +0.0000 USD | +0.00% |
| kling-v2a | $0.1750 | $0.1750 | -0.0000 USD | -0.00% |
| mirelo16 | $0.5100 | $0.5100 | +0.0000 USD | +0.00% |
| sonilo11 | $0.4500 | $0.4289 | +0.0211 USD | +4.93% |

## Rows from other credentials (1)

| CSV row | app id | billed | credential label |
|---:|---|---:|---|
| 2 | fal-ai/hunyuan-video-foley | $0.6586 | API: 92cc21dc... |

## Grand totals

| metric | value |
|---|---:|
| billed total, kept rows (7) | $2.8055 |
| billed total, excluded rows (1) | $0.6586 |
| estimated total (matched endpoints) | $1.9133 |
| delta (billed kept − estimated) | +0.8922 USD |

## Notes

- Credential filter: kept rows whose auth_method contains key id `bfe2a3a9…`; other credentials appear under 'Rows from other credentials'.
- Estimated amounts sum submission-time ledger entries per matched endpoint; they are advisory — billed amounts are the source of truth.
