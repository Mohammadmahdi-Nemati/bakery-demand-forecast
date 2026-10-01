# Ergebnisse (automatisch erzeugt von run.py)

## Genauigkeit der Nachfrageprognose (Testzeitraum, gegen echte Nachfrage)

| Methode | WAPE_% | Bias_% |
|---|---|---|
| Ø letzte 4 gleiche Wochentage (Absatz) | 19,7 | -4,6 |
| Modell ohne Ausverkaufs-Korrektur | 18,9 | -6,3 |
| Modell mit Ausverkaufs-Korrektur | 18,9 | -5,3 |

## Planungsregeln im Vergleich (alle Tage)

| Planung | produziert | abfallquote_% | abfall_kosten_eur | lieferfaehigkeit_% | entgangene_marge_eur | gesamtkosten_eur | gewinn_eur |
|---|---|---|---|---|---|---|---|
| Heutige Planung (Ø 4 Wochen + 12 %) | 269.545 | 13,0 | 14.228 | 92,9 | 18.806 | 33.034 | 186.231 |
| Heutige Planung + Feiertagsregel | 268.914 | 12,4 | 13.498 | 93,3 | 18.410 | 31.908 | 187.357 |
| Modell + Newsvendor-Menge | 267.294 | 12,3 | 14.136 | 92,9 | 17.642 | 31.778 | 187.487 |
| Modell + Newsvendor + Feiertagsregel | 267.322 | 11,9 | 13.770 | 93,3 | 17.149 | 30.918 | 188.347 |

## Nur Feiertage

| Planung | produziert | abfallquote_% | abfall_kosten_eur | lieferfaehigkeit_% | entgangene_marge_eur | gesamtkosten_eur | gewinn_eur |
|---|---|---|---|---|---|---|---|
| Heutige Planung (Ø 4 Wochen + 12 %) | 11.441 | 24,1 | 1.175 | 83,2 | 1.043 | 2.218 | 5.936 |
| Heutige Planung + Feiertagsregel | 10.810 | 10,1 | 445 | 93,0 | 647 | 1.093 | 7.062 |
| Modell + Newsvendor-Menge | 10.782 | 20,4 | 812 | 82,3 | 1.141 | 1.952 | 6.202 |
| Modell + Newsvendor + Feiertagsregel | 10.810 | 10,1 | 445 | 93,0 | 647 | 1.093 | 7.062 |

## Ausverkaufs-Korrektur

- Ausverkaufstage gesamt: 3900
- Absatz unterschätzt die echte Nachfrage an diesen Tagen um 14,8 %
- Nach Korrektur nur noch um 12,2 %
