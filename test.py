from multiprocessing import Pool

def multiply(a):
    return(a)

def main():
    p= Pool(2)
    train = [1,2,3,4,5,6,7,8,9]
    test = [9,8,7,6,5,4,3,2,1]
    params = [(test_sample, train)for test_sample in test]

    a = p.map(multiply,params)
    print(a)

if __name__ == "__main__":
    main()