# Ghidul Cristianei

Tot ce trebuie să știi despre proiect, de la zero: conceptele, ce e configurat și cum se folosește.

---

## 1. Ce construim

Un robot care **sortează singur** cuburi Duplo în „bune” și „defecte”:

```
Camera top vede farfuria cu cuburi
   -> brațul ia un cub (politica VLA)
   -> îl duce și îl rotește în fața camerei de inspecție
   -> Anomalib dă un scor de anomalie
   -> scor mic = bun -> farfuria stângă | scor mare = defect -> farfuria dreaptă
   -> repetă până se golește farfuria
```

La demo-ul final **nimeni nu atinge brațele**. Totul rulează autonom, pe laptopul Intel.

### Cum se dă punctajul
| Criteriu | % | Ce înseamnă |
|---|---|---|
| Task success | 30 | acuratețe, timp per cub, merge și cu cuburi mutate sau în grămadă |
| Integrare Intel | 30 | OpenVINO, modele împărțite pe NPU / GPU / CPU |
| Dataset & pipeline | 20 | demonstrații curate, fine-tuning în Physical AI Studio |
| Eficiență | 20 | latență, temperatură, consum |

---

## 2. Conceptele

### Hardware
- **SO-101**: brațul robotic, cu 6 motoare (bază, umăr, cot, încheietură sus-jos, încheietură rotire, gheară). Starea lui = 6 numere (unghiuri).
- **Leader arm**: brațul pe care îl miști cu mâna. Doar citește unghiuri.
- **Follower arm**: brațul cu motoare puternice. Copiază leader-ul sau rulează singur modelul.
- **Calibrare**: software-ul învață corespondența dintre motoarele leader și follower. Dacă e făcută prost, tot ce urmează iese prost.
- **Camere**: top (RealSense D415, vede masa de sus, dă și adâncime), inspecție (stația unde arătăm cubul), wrist (pe gheară).
- **CPU / GPU / NPU**: cele 3 procesoare din laptopul Intel. NPU = cip dedicat AI, cu consum mic.

### Cum învață robotul
- **Teleoperare (teleop)**: miști leader-ul, follower-ul te copiază. Așa îi arătăm robotului ce să facă.
- **Episod / demonstrație**: o înregistrare a task-ului făcut o dată (imagini + unghiuri, de ~30 de ori pe secundă).
- **Dataset**: toate episoadele adunate.
- **Imitation learning**: robotul învață imitându-ne, nu programându-l pas cu pas.
- **Policy (politică)**: modelul rezultat. Primește imaginile + unghiurile de acum și scoate următoarele unghiuri. Rulează în buclă.
- **VLA (Vision-Language-Action)**: policy care primește și text („pune cubul în stânga”). Același model face lucruri diferite în funcție de instrucțiune.
- **ACT / SmolVLA / Pi0.5**: tipuri de politici din Physical AI Studio. ACT e mic și rapid, fără text; SmolVLA și Pi0.5 sunt VLA-uri.
- **Fine-tuning**: ajustezi un model pre-antrenat pe datele tale. De aceea ajung câteva zeci de demonstrații.
- **Physical AI Studio**: aplicația Intel pentru înregistrare, antrenare în cloud și deploy.

### Cum detectăm defectele
- **Anomaly detection**: modelul vede **doar cuburi bune**, învață cum arată „normal” și dă un scor mare la orice e diferit.
- **Anomalib**: biblioteca Intel pentru asta. Începem cu **PatchCore**.
- **Prag (threshold)**: scor > prag înseamnă defect. Îl calibrăm cu câteva cuburi colorate cu markerul.
- **Re-grasp**: dacă scorul e la limită sau cubul a căzut, robotul îl ia din nou.

### Cum rulează rapid
- **Inferență**: rularea unui model deja antrenat.
- **Latență**: cât durează o inferență.
- **OpenVINO**: convertește modelele ca să ruleze rapid pe CPU / GPU / NPU Intel.
- **Cuantizare INT8**: model de ~4 ori mai mic și mai rapid, cu precizie aproape la fel (unealta NNCF).
- **State machine**: scriptul care ține ordinea fazelor (pick -> inspect -> place).

### Sfaturi pentru demonstrații
- mișcări line, același stil de fiecare dată
- **variază pozițiile cuburilor**
- șterge episoadele ratate (mai bine 40 curate decât 100 murdare)
- **nu muta camerele** după ce ai început să înregistrezi
- lumină constantă

---

## 3. Ce e configurat

### Conectarea la laptopul Intel
De pe laptopul tău: `ssh intel` (configurat în `~/.ssh/config`; pe Windows `%USERPROFILE%\.ssh\config`).
În VS Code: Remote-SSH -> `intel` -> folderul `/home/intel/checkurai`.
În fiecare terminal nou: `conda activate intel_dev_env`.

### Dashboard-ul de camere
1. Pe laptopul tău: `ssh -N -L 8080:localhost:8080 intel` (lasă fereastra deschisă)
2. Deschide http://localhost:8080

Ce vezi: cele 3 camere live, cu `cam X fps · you Y fps · Z ms` (fps-ul camerei, câte cadre ajung la tine, întârzierea).
- **Snapshot** salvează o poză în `~/checkurai/snapshots/<cameră>/` (pentru Anomalib)
- **Închide tab-ul înainte de înregistrare**, altfel camerele sunt ocupate
- Pe Wi-Fi-ul Junction există mereu întârziere. Pentru timp real, deschide pagina pe ecranul laptopului Intel.
- Pornire, dacă s-a oprit: `cd ~/checkurai && nohup python tools/camdash.py > logs/camdash.log 2>&1 &`

### Setări camere
`python tools/v4lctl.py /dev/video0` afișează setările camerei wrist, iar `... nume=valoare` le schimbă.
- Wrist: focus **manual** (inel pe lentilă), fără control din software. Reglat azi.
- Wrist: `exposure_dynamic_framerate=0`, ca să nu scadă la 15 fps.
- Inspecție: are autofocus. Pentru Anomalib trebuie blocat: `focus_automatic_continuous=0 focus_absolute=<valoare>`.
- Setările se pierd dacă scoți camera din USB.

### Git
Repo: https://github.com/bristiana/checkurai (privat, al tău)
- **Regulile sunt obligatorii și verificate automat** (hook-uri + CI): vezi [CONTRIBUTING.md](../CONTRIBUTING.md). Pe scurt: fără secrete (tokenuri, parole, chei), fără date personale (emailuri, telefoane, căi personale), fără dataset-uri, modele sau fișiere mari, doar munca noastră de la eveniment.
- **O singură dată, în fiecare clonă:** `git config core.hooksPath .githooks` și `git config user.email "<id>+<login>@users.noreply.github.com"` (adresa noreply din GitHub → Settings → Emails).
- Flux: `git pull` -> lucrezi -> `git add <fișiere>` (nu `git add -A`) -> `git commit -m "..."` -> `git push`
- Toți lucrăm în **același folder** pe laptop, deci împărțiți-vă pe fișiere.
- **Nu urca dataset-uri sau modele în git** (sunt deja excluse).

---

## 4. Probleme cunoscute

- **3 articulații merg invers** între leader și follower (`elbow_flex`, `wrist_flex`, `wrist_roll`). `teleop_inv.py` le inversează, dar Physical AI Studio și `lerobot-record` nu știu de asta. **Trebuie rezolvat la calibrare înainte de înregistrare.** Întreabă un mentor Intel.
- **Physical AI Studio** nu era pornit la început (vezi `install_studio.sh`).
- **Camera top e întunecată**: trebuie lumină deasupra mesei.
- **Fundalul camerei de inspecție e aglomerat**: pune un carton alb în spate.

---

## 5. Ordinea de lucru

1. Scena fixată: camere, farfurii, lumină, totul lipit cu bandă
2. Articulațiile inversate rezolvate, teleop curat
3. Demonstrații în Physical AI Studio (~50 pick, ~30 stânga, ~30 dreapta)
4. Antrenare policy în cloud; **în paralel** Anomalib pe poze de la camera de inspecție
5. Export OpenVINO: policy pe GPU, Anomalib pe NPU
6. State machine: bucla completă, chiar dacă lentă
7. Abia apoi: INT8, viteză, re-grasp, cuburi în grămadă, dashboard cu metrici

## 6. La final

Vezi [`RETURN_CHECKLIST.md`](../RETURN_CHECKLIST.md): întâi backup, apoi revocare acces și ștergere.
