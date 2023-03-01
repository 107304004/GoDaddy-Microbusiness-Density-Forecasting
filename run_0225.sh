echo '------------------------------'
echo 'BASELINE'
python main.py --cutval_date 2022-12 --shift 0 --cfips 0 --active 0
echo '------------------------------'
echo 'SHIFT ONLY'
python main.py --cutval_date 2022-12 --shift 1 --cfips 0 --active 0
echo '------------------------------'
echo 'CFIPS MEAN ONLY'
python main.py --cutval_date 2022-12 --shift 0 --cfips 1 --active 0
echo '------------------------------'
echo 'CFIPS STD ONLY'
python main.py --cutval_date 2022-12 --shift 0 --cfips 2 --active 0
echo '------------------------------'
echo 'CFIPS TREND ONLY'
python main.py --cutval_date 2022-12 --shift 0 --cfips 3 --active 0
echo '------------------------------'
echo 'CFIPS ONLY'
python main.py --cutval_date 2022-12 --shift 0 --cfips 4 --active 0
echo '------------------------------'
echo 'ACTIVE ONLY'
python main.py --cutval_date 2022-12 --shift 0 --cfips 0 --active 1
echo '------------------------------'
echo 'SHIFT + CFIPS'
python main.py --cutval_date 2022-12 --shift 1 --cfips 4 --active 0
echo '------------------------------'
echo 'SHIFT + ACTIVE'
python main.py --cutval_date 2022-12 --shift 1 --cfips 0 --active 1
echo '------------------------------'
echo 'ALL'
python main.py --cutval_date 2022-12 --shift 1 --cfips 4 --active 1
echo '------------------------------'
