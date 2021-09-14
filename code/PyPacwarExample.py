import _PyPacwar
import numpy


# Example Python module in C for Pacwar
def main():

	test = [int(s) for s in "33333332333230033333330333333330333333331332333333"]
	ones   = [1]*50
	twos   = [2]*50
	threes = [3]*50

	print(len(test))

	(rounds,c1,c2) = _PyPacwar.battle(ones, test)
	print("ones vs test")
	print ("Number of rounds:", rounds )
	print ("Ones PAC-mites remaining:", c1)
	print ("Test PAC-mites remaining:", c2)

	(rounds, c1, c2) = _PyPacwar.battle(twos, test)
	print("twos vs test")
	print("Number of rounds:", rounds)
	print("Twos PAC-mites remaining:", c1)
	print("Test PAC-mites remaining:", c2)

	(rounds, c1, c2) = _PyPacwar.battle(threes, test)
	print("threes vs test")
	print("Number of rounds:", rounds)
	print("Threes PAC-mites remaining:", c1)
	print("Test PAC-mites remaining:", c2)


if __name__ == "__main__": main()
