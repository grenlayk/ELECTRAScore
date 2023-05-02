from analysis import SUMStat

def main():
    summ_stat = SUMStat('data/SummEval/scores.pkl') 
    summ_stat.evaluate_summary('fluency') 

if __name__ == '__main__':
    main()