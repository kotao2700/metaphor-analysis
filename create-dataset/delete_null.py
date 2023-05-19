from settings import TARGET_WORDS

def main():
	for target_word in TARGET_WORDS:
		with open(f'/home/kotaro/work/metaphor-analysis-m1/data/cc100/misnet-input/{target_word}.csv') as f:
			for line in f:
				print(line)
			data = f.read()
			print(data)
		break

if __name__ == "__main__":
    main()
