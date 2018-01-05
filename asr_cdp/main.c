#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asr_cdp_lib.h"

#define TRAIN_MODE 1
#define EVALUATION_MODE 2
#define RECOGNITION_MODE 3

#define NUMBER_OF_ARG_TYPES 7

#define MODEL_NAME 0
#define BASE_DIR 1
#define PART 2
#define JSON 3
#define WORDS 4
#define OUTPUT_FILE 5
#define LIST 6

int parse_cmd(int argc, char* argv[], char* parsed_args[]);
int parse_cmd_for_training(int argc, char* argv[], int used_args[], char* parsed_args[]);
int parse_cmd_for_evaluation(int argc, char* argv[], int used_args[], char* parsed_args[]);
int parse_cmd_for_recognition(int argc, char* argv[], int used_args[], char* parsed_args[]);
int find_arg(int argc, char* argv[], char* found, int used_args[]);

int main(int argc, char* argv[])
{
	TReference* references_for_words;
	TSpectrogram* list_of_spectrograms;
	TTrainDataForWord* spectrograms_of_words_from_json;
	TTrainDataForWord spectrograms_of_silence_from_json;
	int number_of_words_from_json;
	int vocabulary_size, feature_vector_size, tmp_feature_vector_size, number_of_spectrograms;
	float* spectrums_of_silences;
	int number_of_silences;
	char** interesting_words;
	int number_of_interesting_words;
	char* parsed_args[NUMBER_OF_ARG_TYPES];
	int n;
	int mode = parse_cmd(argc, argv, parsed_args);
	if (mode == 0)
	{
		return EXIT_FAILURE;
	}
	if (mode == TRAIN_MODE)
	{
		if (parsed_args[WORDS] != NULL)
		{
			if (!load_interesting_words(parsed_args[WORDS],
				&interesting_words, &number_of_interesting_words))
			{
				fprintf(stderr, "interesting words cannot be loaded from the file `%s`.\n",
					parsed_args[WORDS]);
				return EXIT_FAILURE;
			}
		}
		else
		{
			interesting_words = NULL;
			number_of_interesting_words = 0;
		}
		if (!load_train_data(parsed_args[JSON], parsed_args[BASE_DIR],
			parsed_args[PART], interesting_words, number_of_interesting_words, &feature_vector_size,
			&spectrograms_of_words_from_json, &number_of_words_from_json, &spectrograms_of_silence_from_json))
		{
			fprintf(stderr, "Data for training cannot be loaded by description from `%s`!\n",
				parsed_args[JSON]);
			finalize_interesting_words(interesting_words, number_of_interesting_words);
			return EXIT_FAILURE;
		}
		finalize_interesting_words(interesting_words, number_of_interesting_words);
		spectrums_of_silences = create_references_for_silences(spectrograms_of_silence_from_json, feature_vector_size);
		number_of_silences = spectrograms_of_silence_from_json.n;
		if (spectrums_of_silences == NULL)
		{
			fprintf(stderr, "References for silences cannot be created!\n");
			finalize_train_data(spectrograms_of_words_from_json, number_of_words_from_json);
			finalize_train_data_for_word(spectrograms_of_silence_from_json);
			return EXIT_FAILURE;
		}
		references_for_words = create_references_for_words(spectrograms_of_words_from_json, NULL,
			number_of_words_from_json, feature_vector_size, spectrums_of_silences, number_of_silences,
			20);
		if (references_for_words == NULL)
		{
			fprintf(stderr, "References for words cannot be created!\n");
			free(spectrums_of_silences);
			finalize_train_data(spectrograms_of_words_from_json, number_of_words_from_json);
			finalize_train_data_for_word(spectrograms_of_silence_from_json);
			return EXIT_FAILURE;
		}
		n = save_references(parsed_args[MODEL_NAME], references_for_words, number_of_words_from_json,
			feature_vector_size, spectrums_of_silences, number_of_silences);
		free(spectrums_of_silences);
		finalize_references(references_for_words, number_of_words_from_json);
		finalize_train_data(spectrograms_of_words_from_json, number_of_words_from_json);
		finalize_train_data_for_word(spectrograms_of_silence_from_json);
		if (!n)
		{
			fprintf(stderr, "References for words and silences cannot be saved into the file `%s`.\n",
				parsed_args[MODEL_NAME]);
			return EXIT_FAILURE;
		}
	}
	else if (mode == EVALUATION_MODE)
	{
		if (!load_references(parsed_args[MODEL_NAME], &references_for_words, &vocabulary_size,
			&feature_vector_size, &spectrums_of_silences, &number_of_silences))
		{
			fprintf(stderr, "References cannot be loaded from the file `%s`.\n",
				parsed_args[MODEL_NAME]);
			return EXIT_FAILURE;
		}
		if (!load_train_data(parsed_args[JSON], parsed_args[BASE_DIR],
			parsed_args[PART], NULL, 0, &tmp_feature_vector_size,
			&spectrograms_of_words_from_json, &number_of_words_from_json, &spectrograms_of_silence_from_json))
		{
			fprintf(stderr, "Data for testing cannot be loaded by description from `%s`!\n",
				parsed_args[JSON]);
			finalize_references(references_for_words, vocabulary_size);
			free(spectrums_of_silences);
			return EXIT_FAILURE;
		}
		if (feature_vector_size != tmp_feature_vector_size)
		{
			fprintf(stderr, "Feature vector size of input spectrograms does not correspond to the "\
				"feature vector size of references. %d != %d.\n", feature_vector_size, tmp_feature_vector_size);
			finalize_references(references_for_words, vocabulary_size);
			free(spectrums_of_silences);
			finalize_train_data(spectrograms_of_words_from_json, number_of_words_from_json);
			finalize_train_data_for_word(spectrograms_of_silence_from_json);
			return EXIT_FAILURE;
		}
		n = evaluate(spectrograms_of_words_from_json, number_of_words_from_json, feature_vector_size,
			spectrums_of_silences, number_of_silences, references_for_words, vocabulary_size, 1);
		finalize_references(references_for_words, vocabulary_size);
		free(spectrums_of_silences);
		finalize_train_data(spectrograms_of_words_from_json, number_of_words_from_json);
		finalize_train_data_for_word(spectrograms_of_silence_from_json);
		if (!n)
		{
			return EXIT_FAILURE;
		}
	}
	else
	{
		if (!load_references(parsed_args[MODEL_NAME], &references_for_words, &vocabulary_size,
			&feature_vector_size, &spectrums_of_silences, &number_of_silences))
		{
			fprintf(stderr, "References cannot be loaded from the file `%s`.\n",
				parsed_args[MODEL_NAME]);
			return EXIT_FAILURE;
		}
		if (!load_list_of_spectrograms(parsed_args[LIST], parsed_args[BASE_DIR],
			&list_of_spectrograms, &number_of_spectrograms, &tmp_feature_vector_size))
		{
			fprintf(stderr, "Spectrograms for recognition cannot be loaded from the file `%s`.\n",
				parsed_args[MODEL_NAME]);
			finalize_references(references_for_words, vocabulary_size);
			free(spectrums_of_silences);
			return EXIT_FAILURE;
		}
		if (feature_vector_size != tmp_feature_vector_size)
		{
			fprintf(stderr, "Feature vector size of input spectrograms does not correspond to the "\
				"feature vector size of references. %d != %d.\n", feature_vector_size, tmp_feature_vector_size);
			finalize_references(references_for_words, vocabulary_size);
			free(spectrums_of_silences);
			finalize_spectrograms_list(list_of_spectrograms, number_of_spectrograms);
			return EXIT_FAILURE;
		}
		n = recognize_all(list_of_spectrograms, number_of_spectrograms, feature_vector_size,
			spectrums_of_silences, number_of_silences, references_for_words, vocabulary_size,
			parsed_args[OUTPUT_FILE]);
		finalize_references(references_for_words, vocabulary_size);
		free(spectrums_of_silences);
		finalize_spectrograms_list(list_of_spectrograms, number_of_spectrograms);
		if (n == 0)
		{
			fprintf(stderr, "Input spectrograms cannot be recognized with specified vocabulary of references.\n");
			return EXIT_FAILURE;
		}
		if (n > 1)
		{
			printf("\n%d spectrograms have been successfully recognized.\n", n);
		}
		else
		{
			printf("\nOne spectrogram has been successfully recognized.\n");
		}
	}
	return EXIT_SUCCESS;
}

int parse_cmd(int argc, char* argv[], char* parsed_args[])
{
	int i, j, k, mode = 0;
	int* used_args;
	memset(&parsed_args[0], 0, NUMBER_OF_ARG_TYPES);
	if (argc <= 1)
	{
		return mode;
	}
	used_args = (int*)malloc(sizeof(int) * argc);
	if (used_args == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		return mode;
	}
	memset(used_args, 0, argc);
	used_args[0] = 1;
	i = find_arg(argc, argv, "-t", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--train", used_args);
	}
	j = find_arg(argc, argv, "-e", used_args);
	if (j < 0)
	{
		j = find_arg(argc, argv, "--eval", used_args);
	}
	k = find_arg(argc, argv, "-r", used_args);
	if (k < 0)
	{
		k = find_arg(argc, argv, "--recogn", used_args);
	}
	if ((i < 0) && (j < 0) && (k < 0))
	{
		fprintf(stderr, "Unknown mode!\nThere are three execution modes: training mode "\
			"(\"-t\" or \"--train\"), evaluation mode (\"-e\" or \"--eval\") and "\
			"recognition mode (\"-r\" or \"--recogn\").\n");
		free(used_args);
		return mode;
	}
	if (((i > 0) && (j > 0)) || ((i > 0) && (k > 0)) || ((j > 0) && (k > 0)))
	{
		fprintf(stderr, "Several modes are set at the same time.\nOnly one mode (training, "\
			"evaluation or recognition) can be set.\n");
		free(used_args);
		return mode;
	}
	if (i > 0)
	{
		used_args[i] = 1;
		mode = parse_cmd_for_training(argc, argv, used_args, parsed_args);
	}
	else
	{
		if (j > 0)
		{
			used_args[j] = 1;
			mode = parse_cmd_for_evaluation(argc, argv, used_args, parsed_args);
		}
		else
		{
			if (k > 0)
			{
				used_args[k] = 1;
				mode = parse_cmd_for_recognition(argc, argv, used_args, parsed_args);
			}
		}
	}
	if (mode > 0)
	{
		i = find_arg(argc, argv, "-m", used_args);
		if (i < 0)
		{
			i = find_arg(argc, argv, "--model", used_args);
		}
		if (i < 0)
		{
			mode = 0;
			fprintf(stderr, "The model name (\"-m\" or \"--model\") is not found!\n");
		}
		else
		{
			if (i >= (argc - 1))
			{
				mode = 0;
			}
			else
			{
				if (used_args[i + 1] > 0)
				{
					mode = 0;
				}
			}
			if (mode > 0)
			{
				used_args[i] = 1;
				used_args[i + 1] = 1;
				parsed_args[MODEL_NAME] = argv[i + 1];
			}
			else
			{
				fprintf(stderr, "Value of the model name (\"-m\" or \"--model\") is not specified!\n");
			}
		}
	}
	if (mode > 0)
	{
		i = find_arg(argc, argv, "-d", used_args);
		if (i < 0)
		{
			i = find_arg(argc, argv, "--dir", used_args);
		}
		if (i < 0)
		{
			mode = 0;
			fprintf(stderr, "The base directory with sound data (\"-d\" or \"--dir\") is not found!\n");
		}
		else
		{
			if (i >= (argc - 1))
			{
				mode = 0;
			}
			else
			{
				if (used_args[i + 1] > 0)
				{
					mode = 0;
				}
			}
			if (mode > 0)
			{
				used_args[i] = 1;
				used_args[i + 1] = 1;
				parsed_args[BASE_DIR] = argv[i + 1];
			}
			else
			{
				fprintf(stderr, "Value of the base directory with sound data (\"-d\" or \"--dir\") "\
					"is not specified!\n");
			}
		}
	}
	free(used_args);
	return mode;
}

int find_arg(int argc, char* argv[], char* found, int used_args[])
{
	int i, res = -1;
	if (used_args == NULL)
	{
		for (i = 1; i < argc; ++i)
		{
			if (strcmp(argv[i], found) == 0)
			{
				res = i;
				break;
			}
		}
	}
	else
	{
		for (i = 1; i < argc; ++i)
		{
			if (used_args[i] != 0)
			{
				continue;
			}
			if (strcmp(argv[i], found) == 0)
			{
				res = i;
				break;
			}
		}
	}
	return res;
}

int parse_cmd_for_training(int argc, char* argv[], int used_args[], char* parsed_args[])
{
	int i = find_arg(argc, argv, "-j", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--json", used_args);
	}
	if (i < 0)
	{
		fprintf(stderr, "The JSON file with description of sound files (\"-j\" or \"--json\") "\
			"is not found!\n");
		return 0;
	}
	if (i >= (argc - 1))
	{
		fprintf(stderr, "Value of the JSON file with description of sound files "\
			"(\"-j\" or \"--json\") is not specified!\n");
		return 0;
	}
	if (used_args[i + 1] > 0)
	{
		fprintf(stderr, "Value of the JSON file with description of sound files "\
			"(\"-j\" or \"--json\") is not specified!\n");
		return 0;
	}
	parsed_args[JSON] = argv[i + 1];
	used_args[i] = 1;
	used_args[i + 1] = 1;
	i = find_arg(argc, argv, "-p", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--part", used_args);
	}
	if (i < 0)
	{
		fprintf(stderr, "The used part of sound files (\"-p\" or \"--part\") is not found!");
		return 0;
	}
	if (i >= (argc - 1))
	{
		fprintf(stderr, "Value of the used part of sound files (\"-p\" or \"--part\") "\
			"is not specified!");
		return 0;
	}
	if (used_args[i + 1] > 0)
	{
		fprintf(stderr, "Value of the used part of sound files (\"-p\" or \"--part\") "\
			"is not specified!\n");
		return 0;
	}
	parsed_args[PART] = argv[i + 1];
	used_args[i] = 1;
	used_args[i + 1] = 1;
	i = find_arg(argc, argv, "-w", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--word", used_args);
	}
	if (i < 0)
	{
		parsed_args[WORDS] = NULL;
	}
	else
	{
		if (i >= (argc - 1))
		{
			fprintf(stderr, "Value of the interesting words list (\"-w\" or \"--words\") "\
				"is not specified!\n");
			return 0;
		}
		if (used_args[i + 1] > 0)
		{
			fprintf(stderr, "Value of the interesting words list (\"-w\" or \"--words\") "\
				"is not specified!\n");
			return 0;
		}
		parsed_args[WORDS] = argv[i + 1];
		used_args[i] = 1;
		used_args[i + 1] = 1;
	}
	return TRAIN_MODE;
}

int parse_cmd_for_evaluation(int argc, char* argv[], int used_args[], char* parsed_args[])
{
	int i = find_arg(argc, argv, "-j", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--json", used_args);
	}
	if (i < 0)
	{
		fprintf(stderr, "The JSON file with description of sound files (\"-j\" or \"--json\") "\
			"is not found!\n");
		return 0;
	}
	if (i >= (argc - 1))
	{
		fprintf(stderr, "Value of the JSON file with description of sound files "\
			"(\"-j\" or \"--json\") is not specified!\n");
		return 0;
	}
	if (used_args[i + 1] > 0)
	{
		fprintf(stderr, "Value of the JSON file with description of sound files "\
			"(\"-j\" or \"--json\") is not specified!\n");
		return 0;
	}
	parsed_args[JSON] = argv[i + 1];
	used_args[i] = 1;
	used_args[i + 1] = 1;
	i = find_arg(argc, argv, "-p", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--part", used_args);
	}
	if (i < 0)
	{
		fprintf(stderr, "The used part of sound files (\"-p\" or \"--part\") is not found!\n");
		return 0;
	}
	if (i >= (argc - 1))
	{
		fprintf(stderr, "Value of the used part of sound files (\"-p\" or \"--part\") "\
			"is not specified!\n");
		return 0;
	}
	if (used_args[i + 1] > 0)
	{
		fprintf(stderr, "Value of the used part of sound files (\"-p\" or \"--part\") "\
			"is not specified!\n");
		return 0;
	}
	parsed_args[PART] = argv[i + 1];
	used_args[i] = 1;
	used_args[i + 1] = 1;
	return EVALUATION_MODE;
}

int parse_cmd_for_recognition(int argc, char* argv[], int used_args[], char* parsed_args[])
{
	int i = find_arg(argc, argv, "-l", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--list", used_args);
	}
	if (i < 0)
	{
		fprintf(stderr, "The sounds list for recognition (\"-l\" or \"--list\") is not found!\n");
		return 0;
	}
	if (i >= (argc - 1))
	{
		fprintf(stderr, "Value of the sounds list for recognition (\"-l\" or \"--list\") "\
			"is not specified!\n");
		return 0;
	}
	if (used_args[i + 1] > 0)
	{
		fprintf(stderr, "Value of the sounds list for recognition (\"-l\" or \"--list\") "\
			"is not specified!\n");
		return 0;
	}
	parsed_args[LIST] = argv[i + 1];
	used_args[i] = 1;
	used_args[i + 1] = 1;
	i = find_arg(argc, argv, "-o", used_args);
	if (i < 0)
	{
		i = find_arg(argc, argv, "--output", used_args);
	}
	if (i < 0)
	{
		parsed_args[OUTPUT_FILE] = NULL;
	}
	else
	{
		if (i >= (argc - 1))
		{
			fprintf(stderr, "Value of the output file for recognition results (\"-o\" or \"--output\") "\
				"is not specified!\n");
			return 0;
		}
		if (used_args[i + 1] > 0)
		{
			fprintf(stderr, "Value of the output file for recognition results (\"-o\" or \"--output\") "\
				"is not specified!\n");
			return 0;
		}
		parsed_args[OUTPUT_FILE] = argv[i + 1];
		used_args[i] = 1;
		used_args[i + 1] = 1;
	}
	return RECOGNITION_MODE;
}
