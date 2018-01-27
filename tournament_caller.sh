printf '\n_____________________________________________________________________\n' >> results.txt
for VARIABLE in 1 2 3 4 5
do
    python3 tournament.py >> results.txt
done
