# Manuelle Abnahmefälle – Missionsorchestrierung

## 1) Happy Path mit 3 Punkten
- **Voraussetzungen:**
  - Gültige Mission mit 3 Punkten.
  - Robotik-Navigation erreichbar, Messdienst liefert valide Messresultate.
- **Schritte:**
  1. Mission laden.
  2. Lauf starten.
  3. Warten bis alle 3 Punkte abgearbeitet sind.
- **Erwartetes Ergebnis:**
  - Navigation für alle Punkte erfolgreich.
  - Pro Punkt genau ein Messresultat.
  - Pro Punkt genau eine LIDAR-Referenzmessung über `ros2 topic echo /scan --once`.
  - Run-Status `completed`, `succeeded_points=3`, `failed_points=0`.
  - Drei Punkt-Logs, drei `.lidar.scan.txt` Referenzdateien und ein `run-summary.json` vorhanden.

## 2) Navigationsabbruch am 2. Punkt
- **Voraussetzungen:**
  - Mission mit 3 Punkten.
  - Für Punkt 2 liefert Adapter `aborted`.
- **Schritte:**
  1. Lauf mit `on_point_error=continue` starten.
  2. Ablauf bis Punkt 3 durchlaufen lassen.
- **Erwartetes Ergebnis:**
  - Punkt 1 erfolgreich.
  - Punkt 2 mit Fehlerstatus `navigation_failed:aborted`.
  - Punkt 3 wird trotzdem ausgeführt.
  - Run endet `completed` mit genau 1 fehlgeschlagenem Punkt.

## 3) Messfehler bei erfolgreicher Navigation
- **Voraussetzungen:**
  - Navigation aller Punkte erfolgreich.
  - Messdienst wirft am Zielpunkt (z. B. Punkt 2) einen Fehler.
- **Schritte:**
  1. Lauf mit `on_point_error=continue` starten.
  2. Bis Laufende laufen lassen.
- **Erwartetes Ergebnis:**
  - Betroffener Punkt wird als `failed` protokolliert.
  - Fehlertext aus Messdienst erscheint im Punkt-Log.
  - Nachfolgende Punkte werden weiter verarbeitet.
  - Run-Summary zählt den Messfehler als fehlgeschlagenen Punkt.

## 4) Stop/Resume mitten im Lauf
- **Voraussetzungen:**
  - Mission mit mindestens 3 Punkten.
  - Lauf kann pausiert und wieder aufgenommen werden.
- **Schritte:**
  1. Lauf starten.
  2. Während der Abarbeitung `Pause` auslösen.
  3. `Resume` auslösen.
  4. Kurz danach `Stop` auslösen.
- **Erwartetes Ergebnis:**
  - Zustandswechsel `running -> paused -> running -> stopping -> completed`.
  - Aktiver Navigationsauftrag wird abgebrochen (`cancel_current_goal`).
  - Run-Summary enthält `abort_reason=stopped`.
  - Bereits persistierte Punkt-Logs bleiben konsistent erhalten.
