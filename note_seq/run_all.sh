#!/bin/bash
set -e

python end2end.py --experiment led_v2 -baseline -debug
python end2end.py --experiment led_v2 -baseline -empty_init -debug
python end2end.py --experiment led_v2 -debug
python end2end.py --experiment led_v2 -empty_init -debug
python end2end.py --experiment led_v2 -oracle_init -debug
python end2end.py --experiment led_v2 -rejection -debug
python end2end.py --experiment led_v2 -empty_init -rejection -debug
python end2end.py --experiment led_v2 -oracle_init -rejection -debug

python end2end.py --experiment led_v2 -baseline
python end2end.py --experiment led_v2 -baseline -empty_init
python end2end.py --experiment led_v2
python end2end.py --experiment led_v2 -empty_init
python end2end.py --experiment led_v2 -oracle_init
python end2end.py --experiment led_v2 -rejection
python end2end.py --experiment led_v2 -empty_init -rejection
python end2end.py --experiment led_v2 -oracle_init -rejection

python end2end.py --experiment note_seq_v2 -baseline
python end2end.py --experiment note_seq_v2 -baseline -empty_init
python end2end.py --experiment note_seq_v2
python end2end.py --experiment note_seq_v2 -empty_init
python end2end.py --experiment note_seq_v2 -oracle_init
python end2end.py --experiment note_seq_v2 -rejection
python end2end.py --experiment note_seq_v2 -empty_init -rejection
python end2end.py --experiment note_seq_v2 -oracle_init -rejection


python end2end.py --experiment led_v2 --note_window 5
python end2end.py --experiment led_v2 -empty_init --note_window 5
python end2end.py --experiment led_v2 -oracle_init --note_window 5
python end2end.py --experiment led_v2 -rejection --note_window 5
python end2end.py --experiment led_v2 -empty_init -rejection --note_window 5
python end2end.py --experiment led_v2 -oracle_init -rejection --note_window 5

python end2end.py --experiment note_seq_v2
python end2end.py --experiment note_seq_v2 -empty_init
python end2end.py --experiment note_seq_v2 -oracle_init
python end2end.py --experiment note_seq_v2 -rejection
python end2end.py --experiment note_seq_v2 -empty_init -rejection
python end2end.py --experiment note_seq_v2 -oracle_init -rejection
