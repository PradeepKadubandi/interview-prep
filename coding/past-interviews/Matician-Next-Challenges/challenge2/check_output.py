import sys

def main(e_file, a_file):
    l_no = 0
    with open(e_file, 'r') as ef:
        with open(a_file, 'r') as af:
            while True:
                el = ef.readline().strip()
                al = af.readline().strip()
                l_no += 1
                if len(el.strip()) == 0:
                    print ('Finished Reading Expected File')
                    if len(al.strip()) > 0:
                        print ('Failure: Still there are characters left in Actual File.')
                    return
                if el != al:
                    print ('Line {} differ:'.format(l_no))
                    print ('Expected : {}'.format(el))
                    print ('Actual   : {}'.format(al))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ('Usage : {} expected_output_file actual_output_file'.format(__file__))
    else:
        main(*sys.argv[1:])