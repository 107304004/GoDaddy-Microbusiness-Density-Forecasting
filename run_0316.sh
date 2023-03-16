echo '------------------------------'
echo 'BASELINE'
python main.py --cutval_date 2022-12 --shift 0 --cfips 0 --active 0
echo '------------------------------'
echo 'SHIFT ONLY'
python main.py --cutval_date 2022-12 --shift 1 --cfips 0 --active 0
echo '------------------------------'
echo 'SHIFT + CFIPS'
python main.py --cutval_date 2022-12 --shift 1 --cfips 1 --active 0
echo '------------------------------'
echo 'SHIFT + ACTIVE'
python main.py --cutval_date 2022-12 --shift 1 --cfips 0 --active 1
echo '------------------------------'
echo 'ALL'
python main.py --cutval_date 2022-12 --shift 1 --cfips 1 --active 1
echo '------------------------------'
