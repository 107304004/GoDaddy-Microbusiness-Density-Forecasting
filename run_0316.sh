echo '------------------------------'
echo 'BASELINE'
python main.py --cutval_date 2022-12 --shift 0 --cfips 0 --active 0
echo '------------------------------'
echo 'LAGS ONLY'
python main.py --cutval_date 2022-12 --shift 1 --cfips 0 --active 0
echo '------------------------------'
echo 'LAGS + CFIPS'
python main.py --cutval_date 2022-12 --shift 1 --cfips 1 --active 0
echo '------------------------------'
echo 'LAGS + ACTIVE'
python main.py --cutval_date 2022-12 --shift 1 --cfips 0 --active 1
echo '------------------------------'
echo 'ALL'
python main.py --cutval_date 2022-12 --shift 1 --cfips 1 --active 1
echo '------------------------------'
