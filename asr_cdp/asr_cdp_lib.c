#include <errno.h>
#include <float.h>
#include <math.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "json.h"
#include "asr_cdp_lib.h"

#define BUFFER_SIZE 16384

#ifdef _WIN32
#define DIRCHAR '\\'
#else
#define DIRCHAR '/'
#endif // WINDOWS


float calculate_similarity(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float reference_spectrum[])
{
	int i, j, start_pos = 0;
	float res = 0.0, instant_res, tmp;
	for (i = 0; i < spectrogram_size; ++i)
	{
		instant_res = 0.0;
		for (j = 0; j < feature_vector_size; ++j)
		{
			tmp = spectrogram[start_pos++] - reference_spectrum[j];
			instant_res += tmp * tmp;
		}
		res -= sqrt(instant_res);
	}
	return res;
}

float find_reference_spectrum(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float reference_spectrum[], float tmp_dist_matrix[])
{
	int i, j, start_pos_1, start_pos_2, spectrum_ind, reference_spectrum_ind = 0;
	float res, best_res, dist, tmp;
	tmp_dist_matrix[0] = 0.0;
	best_res = 0.0;
	for (i = 1, start_pos_2 = feature_vector_size; i < spectrogram_size; ++i, start_pos_2 += feature_vector_size)
	{
		dist = 0.0;
		for (j = 0; j < feature_vector_size; ++j)
		{
			tmp = spectrogram[start_pos_2 + j] - spectrogram[j];
			dist += tmp * tmp;
		}
		dist = sqrt(dist);
		tmp_dist_matrix[i] = dist;
		best_res -= dist;
	}
	for (spectrum_ind = 1; spectrum_ind < spectrogram_size; ++spectrum_ind)
	{
		res = 0.0;
		start_pos_1 = spectrum_ind * feature_vector_size;
		for (i = 0, start_pos_2 = 0; i < spectrogram_size; ++i, start_pos_2 += feature_vector_size)
		{
			if (i == spectrum_ind)
			{
				tmp_dist_matrix[spectrum_ind * spectrogram_size + i] = 0.0;
				dist = 0.0;
			}
			else
			{
				if (i < spectrum_ind)
				{
					dist = tmp_dist_matrix[i * spectrogram_size + spectrum_ind];
					tmp_dist_matrix[spectrum_ind * spectrogram_size + i] = dist;
				}
				else
				{
					dist = 0.0;
					for (j = 0; j < feature_vector_size; ++j)
					{
						tmp = spectrogram[start_pos_2 + j] - spectrogram[j + start_pos_1];
						dist += tmp * tmp;
					}
					dist = sqrt(dist);
					tmp_dist_matrix[spectrum_ind * spectrogram_size + i] = dist;
				}
			}
			res -= dist;
		}
		if (res > best_res)
		{
			best_res = res;
			reference_spectrum_ind = spectrum_ind;
		}
	}
	if (reference_spectrum != NULL)
	{
		start_pos_1 = reference_spectrum_ind * feature_vector_size;
		for (i = 0; i < feature_vector_size; ++i)
		{
			reference_spectrum[i] = spectrogram[start_pos_1 + i];
		}
	}
	return best_res;
}

int recognize_one_sound(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float silence_spectrums[], int number_of_silences,
	TReference references[], int vocabulary_size,
	float best_similarities[], float similarities[], float dp_matrix[])
{
	int word_index, silence_index, time_index, state_index, states_number;
	int m, M, u;
	float val, F;
	int best_word_index = -1;
	int ok;
	float best_word_similarity = -FLT_MAX;
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		best_similarities[word_index] = -FLT_MAX;
	}
	for (silence_index = 0; silence_index < number_of_silences; ++silence_index)
	{
		for (word_index = 0; word_index < vocabulary_size; ++word_index)
		{
			states_number = 2 + references[word_index].n;
			for (time_index = 0; time_index < spectrogram_size; ++time_index)
			{
				state_index = 0;
				dp_matrix[time_index * states_number + state_index] =
					((time_index > 0) ? dp_matrix[(time_index - 1) * states_number + state_index] : 0.0) +
					calculate_similarity(&spectrogram[time_index * feature_vector_size], 1, feature_vector_size,
						&silence_spectrums[silence_index * feature_vector_size]);
				for (state_index = 1; state_index < (states_number - 1); ++state_index)
				{
					m = references[word_index].reference[state_index - 1].m;
					M = references[word_index].reference[state_index - 1].M;
					if (((time_index - m) == -1) && (state_index == 1))
					{
						F = 0.0;
					}
					else
					{
						if ((time_index - m) < 0)
						{
							F = -FLT_MAX;
						}
						else
						{
							F = dp_matrix[(time_index - m) * states_number + state_index - 1];
						}
					}
					if (F > (-FLT_MAX / 2.0))
					{
						dp_matrix[time_index * states_number + state_index] = F + calculate_similarity(
							&spectrogram[(time_index - m + 1) * feature_vector_size],
							m, feature_vector_size,
							references[word_index].reference[state_index - 1].spectrum);
						for (u = m + 1; u <= M; u++)
						{
							if (((time_index - u) == -1) && (state_index == 1))
							{
								F = 0.0;
							}
							else
							{
								if ((time_index - u) < 0)
								{
									F = -FLT_MAX;
								}
								else
								{
									F = dp_matrix[(time_index - u) * states_number + state_index - 1];
								}
							}
							if (F > (-FLT_MAX / 2.0))
							{
								val = F + calculate_similarity(
									&spectrogram[(time_index - u + 1) * feature_vector_size],
									u, feature_vector_size,
									references[word_index].reference[state_index - 1].spectrum);
								if (val > dp_matrix[time_index * states_number + state_index])
								{
									dp_matrix[time_index * states_number + state_index] = val;
								}
							}
						}
					}
					else
					{
						dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
					}
				}
				state_index = states_number - 1;
				F = (time_index > 0) ? dp_matrix[(time_index - 1) * states_number + state_index] :
					-FLT_MAX;
				if (F > (-FLT_MAX / 2.0))
				{
					dp_matrix[time_index * states_number + state_index] = F + calculate_similarity(
						&spectrogram[time_index * feature_vector_size], 1, feature_vector_size,
						&silence_spectrums[silence_index * feature_vector_size]);
				}
				else
				{
					dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
				}
				if (dp_matrix[time_index * states_number + state_index - 1] >
					dp_matrix[time_index * states_number + state_index])
				{
					dp_matrix[time_index * states_number + state_index] =
						dp_matrix[time_index * states_number + state_index - 1];
				}
			}
			time_index = spectrogram_size - 1;
			state_index = states_number - 1;
			similarities[word_index] = dp_matrix[time_index * states_number + state_index];
		}
		ok = 0;
		for (word_index = 0; word_index < vocabulary_size; ++word_index)
		{
			if (similarities[word_index] > best_word_similarity)
			{
				best_word_index = word_index;
				best_word_similarity = similarities[word_index];
				ok = 1;
			}
		}
		if (ok > 0)
		{
			for (word_index = 0; word_index < vocabulary_size; ++word_index)
			{
				best_similarities[word_index] = similarities[word_index];
				if (best_similarities[word_index] > (-FLT_MAX / 2))
				{
					best_similarities[word_index] /= spectrogram_size;
				}
			}
		}
	}
	return best_word_index;
}

int recognize_all(TSpectrogram spectrograms[], int spectrograms_number,
	int feature_vector_size, float silence_spectrums[], int number_of_silences,
	TReference references_of_words[], int vocabulary_size, char* output_file)
{
	int i, j, predicted_index, n;
	float *dp_matrix, *similarities, *best_similarities;
	int max_spectrogram_size, max_number_of_states;
	FILE *fp;
	if ((output_file == NULL) || (strlen(output_file) == 0))
	{
		fp = stdout;
	}
	else
	{
		fp = fopen(output_file, "w");
		if (fp == NULL)
		{
			fprintf(stderr, "File `%s` cannot be opened for writing!\n%s\n", output_file, strerror(errno));
			return 0;
		}
	}
	max_spectrogram_size = spectrograms[0].n;
	for (i = 1; i < spectrograms_number; ++i)
	{
		if (spectrograms[i].n > max_spectrogram_size)
		{
			max_spectrogram_size = spectrograms[i].n;
		}
	}
	max_number_of_states = references_of_words[0].n;
	fprintf(fp, "%s", references_of_words[0].wordname);
	for (i = 1; i < vocabulary_size; ++i)
	{
		if (references_of_words[i].n > max_number_of_states)
		{
			max_number_of_states = references_of_words[i].n;
		}
		fprintf(fp, ",%s", references_of_words[i].wordname);
	}
	fprintf(fp, "\n");
	max_number_of_states += 2;
	dp_matrix = (float*)malloc(max_spectrogram_size * max_number_of_states * sizeof(float));
	similarities = (float*)malloc(feature_vector_size * sizeof(float));
	best_similarities = (float*)malloc(feature_vector_size * sizeof(float));
	if ((dp_matrix == NULL) || (similarities == NULL) || (best_similarities == NULL))
	{
		if (dp_matrix != NULL)
		{
			free(dp_matrix);
		}
		if (similarities != NULL)
		{
			free(similarities);
		}
		if (best_similarities != NULL)
		{
			free(best_similarities);
		}
		fprintf(stderr, "Out of memory!\n");
		return 0;
	}
	n = 0;
	for (i = 0; i < spectrograms_number; ++i)
	{
		predicted_index = recognize_one_sound(spectrograms[i].spectrogram, spectrograms[i].n, feature_vector_size,
			silence_spectrums, number_of_silences, references_of_words, vocabulary_size,
			best_similarities, similarities, dp_matrix);
		if (best_similarities[predicted_index] > (-FLT_MAX / 2.0))
		{
			n += 1;
			if (best_similarities[0] > (-FLT_MAX / 2.0))
			{
				fprintf(fp, "%f", best_similarities[0]);
			}
			else
			{
				fprintf(fp, "-Inf");
			}
			for (j = 1; j < vocabulary_size; ++j)
			{
				if (best_similarities[j] >(-FLT_MAX / 2.0))
				{
					fprintf(fp, ",%f", best_similarities[j]);
				}
				else
				{
					fprintf(fp, ",-Inf");
				}
			}
		}
		else
		{
			fprintf(fp, "-Inf");
			for (j = 1; j < vocabulary_size; ++j)
			{
				fprintf(fp, ",-Inf");
			}
		}
		fprintf(fp, "\n");
	}
	free(dp_matrix);
	free(similarities);
	free(best_similarities);
	if (output_file == NULL)
	{
		fclose(fp);
	}
	return n;
}

float do_segmentation(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float silence_spectrums[], int number_of_silences,
	TReference reference, int lengths_of_segments[],
	float dp_matrix[], int dp_matrix_for_lengths[])
{
	int silence_index, time_index, state_index, states_number;
	int m, M, u;
	float val, F;
	float similarity, best_similarity = -FLT_MAX;
	for (silence_index = 0; silence_index < number_of_silences; ++silence_index)
	{
		states_number = 2 + reference.n;
		for (time_index = 0; time_index < spectrogram_size; ++time_index)
		{
			state_index = 0;
			dp_matrix[time_index * states_number + state_index] =
				((time_index > 0) ? dp_matrix[(time_index - 1) * states_number + state_index] : 0.0) +
				calculate_similarity(&spectrogram[time_index * feature_vector_size], 1, feature_vector_size,
					&silence_spectrums[silence_index * feature_vector_size]);
			dp_matrix_for_lengths[time_index * states_number + state_index] = time_index + 1;
			for (state_index = 1; state_index < (states_number - 1); ++state_index)
			{
				m = reference.reference[state_index - 1].m;
				M = reference.reference[state_index - 1].M;
				if (((time_index - m) == -1) && (state_index == 1))
				{
					F = 0.0;
				}
				else
				{
					if ((time_index - m) < 0)
					{
						F = -FLT_MAX;
					}
					else
					{
						F = dp_matrix[(time_index - m) * states_number + state_index - 1];
					}
				}
				dp_matrix_for_lengths[time_index * states_number + state_index] = m;
				if (F > (-FLT_MAX / 2.0))
				{
					dp_matrix[time_index * states_number + state_index] = F + calculate_similarity(
						&spectrogram[(time_index - m + 1) * feature_vector_size],
						m, feature_vector_size,
						reference.reference[state_index - 1].spectrum);
					for (u = m + 1; u <= M; u++)
					{
						if (((time_index - u) == -1) && (state_index == 1))
						{
							F = 0.0;
						}
						else
						{
							if ((time_index - u) < 0)
							{
								F = -FLT_MAX;
							}
							else
							{
								F = dp_matrix[(time_index - u) * states_number + state_index - 1];
							}
						}
						if (F > (-FLT_MAX / 2.0))
						{
							val = F + calculate_similarity(
								&spectrogram[(time_index - u + 1) * feature_vector_size],
								u, feature_vector_size,
								reference.reference[state_index - 1].spectrum);
							if (val > dp_matrix[time_index * states_number + state_index])
							{
								dp_matrix[time_index * states_number + state_index] = val;
								dp_matrix_for_lengths[time_index * states_number + state_index] = u;
							}
						}
					}
				}
				else
				{
					dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
				}
			}
			state_index = states_number - 1;
			F = (time_index > 0) ? dp_matrix[(time_index - 1) * states_number + state_index] :
				-FLT_MAX;
			if (F > (-FLT_MAX / 2.0))
			{
				dp_matrix[time_index * states_number + state_index] = F + calculate_similarity(
					&spectrogram[time_index * feature_vector_size], 1, feature_vector_size,
					&silence_spectrums[silence_index * feature_vector_size]);
				dp_matrix_for_lengths[time_index * states_number + state_index] = 
					dp_matrix_for_lengths[(time_index - 1) * states_number + state_index] + 1;
			}
			else
			{
				dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
				dp_matrix_for_lengths[time_index * states_number + state_index] = 0;
			}
			if (dp_matrix[time_index * states_number + state_index - 1] >=
				dp_matrix[time_index * states_number + state_index])
			{
				dp_matrix[time_index * states_number + state_index] =
					dp_matrix[time_index * states_number + state_index - 1];
				dp_matrix_for_lengths[time_index * states_number + state_index] = 0;
			}
		}
		time_index = spectrogram_size - 1;
		state_index = states_number - 1;
		similarity = dp_matrix[time_index * states_number + state_index];
		if (similarity > best_similarity)
		{
			best_similarity = similarity;
			state_index = states_number - 1;
			time_index = spectrogram_size - 1;
			lengths_of_segments[state_index] = 
				dp_matrix_for_lengths[time_index * states_number + state_index];
			for (state_index = states_number - 2; state_index >= 0; --state_index)
			{
				time_index -= lengths_of_segments[state_index + 1];
				lengths_of_segments[state_index] =
					dp_matrix_for_lengths[time_index * states_number + state_index];
			}
		}
	}
	return best_similarity / spectrogram_size;
}

float do_self_segmentation(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float silence_spectrums[], int number_of_silences, int speech_segments_number,
	int lengths_of_segments[], float dp_matrix[], int dp_matrix_for_lengths[], float tmp_dist_matrix[])
{
	int silence_index, time_index, state_index, states_number;
	int m, M, u;
	float val, F;
	float similarity, best_similarity = -FLT_MAX;
	m = 1;
	M = spectrogram_size - 2;
	states_number = 2 + speech_segments_number;
	for (silence_index = 0; silence_index < number_of_silences; ++silence_index)
	{
		for (time_index = 0; time_index < spectrogram_size; ++time_index)
		{
			state_index = 0;
			if ((time_index + 1) > M)
			{
				dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
				dp_matrix_for_lengths[time_index * states_number + state_index] = 0;
			}
			else
			{
				dp_matrix[time_index * states_number + state_index] = calculate_similarity(
					spectrogram, time_index + 1, feature_vector_size,
					&silence_spectrums[silence_index * feature_vector_size]);
				dp_matrix_for_lengths[time_index * states_number + state_index] = time_index + 1;
			}
			for (state_index = 1; state_index < (states_number - 1); ++state_index)
			{
				if ((time_index - m) < 0)
				{
					F = -FLT_MAX;
				}
				else
				{
					F = dp_matrix[(time_index - m) * states_number + state_index - 1];
				}
				if (F > (-FLT_MAX / 2.0))
				{
					dp_matrix_for_lengths[time_index * states_number + state_index] = m;
					dp_matrix[time_index * states_number + state_index] = F + find_reference_spectrum(
						&spectrogram[(time_index - m + 1) * feature_vector_size],
						m, feature_vector_size, NULL, tmp_dist_matrix);
					for (u = m + 1; u <= M; u++)
					{
						if ((time_index - u) < 0)
						{
							F = -FLT_MAX;
						}
						else
						{
							F = dp_matrix[(time_index - u) * states_number + state_index - 1];
						}
						if (F > (-FLT_MAX / 2.0))
						{
							val = F + find_reference_spectrum(
								&spectrogram[(time_index - u + 1) * feature_vector_size],
								u, feature_vector_size, NULL, tmp_dist_matrix);
							if (val > dp_matrix[time_index * states_number + state_index])
							{
								dp_matrix[time_index * states_number + state_index] = val;
								dp_matrix_for_lengths[time_index * states_number + state_index] = u;
							}
						}
					}
				}
				else
				{
					dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
					dp_matrix_for_lengths[time_index * states_number + state_index] = 0;
				}
			}
			state_index = states_number - 1;
			if ((time_index - m) < 0)
			{
				F = -FLT_MAX;
			}
			else
			{
				F = dp_matrix[(time_index - m) * states_number + state_index - 1];
			}
			if (F > (-FLT_MAX / 2.0))
			{
				dp_matrix_for_lengths[time_index * states_number + state_index] = m;
				dp_matrix[time_index * states_number + state_index] = F + calculate_similarity(
					&spectrogram[(time_index - m + 1) * feature_vector_size], m, feature_vector_size,
					&silence_spectrums[silence_index * feature_vector_size]);
				for (u = m + 1; u <= M; u++)
				{
					if ((time_index - u) < 0)
					{
						F = -FLT_MAX;
					}
					else
					{
						F = dp_matrix[(time_index - u) * states_number + state_index - 1];
					}
					if (F > (-FLT_MAX / 2.0))
					{
						val = F + calculate_similarity(
							&spectrogram[(time_index - u + 1) * feature_vector_size], u,
							feature_vector_size, &silence_spectrums[silence_index * feature_vector_size]);
						if (val > dp_matrix[time_index * states_number + state_index])
						{
							dp_matrix[time_index * states_number + state_index] = val;
							dp_matrix_for_lengths[time_index * states_number + state_index] = u;
						}
					}
				}
			}
			else
			{
				dp_matrix[time_index * states_number + state_index] = -FLT_MAX;
				dp_matrix_for_lengths[time_index * states_number + state_index] = 0;
			}
		}
		time_index = spectrogram_size - 1;
		state_index = states_number - 1;
		similarity = dp_matrix[time_index * states_number + state_index];
		if (similarity > best_similarity)
		{
			best_similarity = similarity;
			state_index = states_number - 1;
			time_index = spectrogram_size - 1;
			lengths_of_segments[state_index] =
				dp_matrix_for_lengths[time_index * states_number + state_index];
			for (state_index = states_number - 2; state_index >= 0; --state_index)
			{
				time_index -= lengths_of_segments[state_index + 1];
				lengths_of_segments[state_index] =
					dp_matrix_for_lengths[time_index * states_number + state_index];
			}
		}
	}
	return best_similarity / spectrogram_size;
}

int evaluate(TTrainDataForWord test_data[], int all_words_number,
	int feature_vector_size, float silence_spectrums[], int number_of_silences,
	TReference references_of_words[], int vocabulary_size, int print_confusion_matrix)
{
	int ok = 1;
	int i, j, k, true_index, predicted_index;
	int corrects = 0, all = 0;
	int max_spectrogram_size = 0, max_states_number = 0;
	float accuracy;
	int *confusion_matrix;
	float *dp_matrix, *similarities, *best_similarities;
	char **interesting_words = (char**)malloc(vocabulary_size * sizeof(char*));
	if (interesting_words == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		return 0;
	}
	for (i = 0; i < vocabulary_size; ++i)
	{
		interesting_words[i] = NULL;
	}
	for (i = 0; i < vocabulary_size; ++i)
	{
		j = strlen(references_of_words[i].wordname);
		interesting_words[i] = (char*)malloc(j * sizeof(char));
		if (interesting_words[i] == NULL)
		{
			ok = 0;
			break;
		}
		memset(interesting_words[i], 0, j * sizeof(char));
		strcpy(interesting_words[i], references_of_words[i].wordname);
		if (references_of_words[i].n > max_states_number)
		{
			max_states_number = references_of_words[i].n;
		}
	}
	if (!ok)
	{
		fprintf(stderr, "Out of memory!\n");
		finalize_interesting_words(interesting_words, vocabulary_size);
		return 0;
	}
	max_states_number += 2;
	j = 0;
	for (i = 0; i < all_words_number; ++i)
	{
		if (find_word(test_data[i].wordname, interesting_words, vocabulary_size) >= 0)
		{
			j += 1;
			for (k = 0; k < test_data[i].n; ++k)
			{
				if (test_data[i].spectrograms[k].n > max_spectrogram_size)
				{
					max_spectrogram_size = test_data[i].spectrograms[k].n;
				}
			}
		}
	}
	if (j == 0)
	{
		finalize_interesting_words(interesting_words, vocabulary_size);
		fprintf(stderr, "There are no interesting words in data for testing!\n");
		return 0;
	}
	confusion_matrix = (int*)malloc(vocabulary_size * vocabulary_size * sizeof(int));
	if (confusion_matrix == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		finalize_interesting_words(interesting_words, vocabulary_size);
		return 0;
	}
	memset(confusion_matrix, 0, vocabulary_size * vocabulary_size * sizeof(int));
	dp_matrix = (float*)malloc(max_states_number * max_spectrogram_size * sizeof(float));
	if (dp_matrix == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		finalize_interesting_words(interesting_words, vocabulary_size);
		free(confusion_matrix);
		return 0;
	}
	similarities = (float*)malloc(feature_vector_size * sizeof(float));
	if (similarities == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		finalize_interesting_words(interesting_words, vocabulary_size);
		free(confusion_matrix);
		free(dp_matrix);
		return 0;
	}
	best_similarities = (float*)malloc(feature_vector_size * sizeof(float));
	if (best_similarities == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		finalize_interesting_words(interesting_words, vocabulary_size);
		free(confusion_matrix);
		free(dp_matrix);
		free(similarities);
		return 0;
	}
	for (i = 0; i < all_words_number; ++i)
	{
		true_index = find_word(test_data[i].wordname, interesting_words, vocabulary_size);
		if (true_index >= 0)
		{
			for (j = 0; j < test_data[i].n; ++j)
			{
				predicted_index = recognize_one_sound(test_data[i].spectrograms[j].spectrogram,
					test_data[i].spectrograms[j].n, feature_vector_size,
					silence_spectrums, number_of_silences, references_of_words, vocabulary_size,
					best_similarities, similarities, dp_matrix);
				confusion_matrix[true_index * vocabulary_size + predicted_index] += 1;
			}
		}
	}
	for (i = 0; i < vocabulary_size; ++i)
	{
		corrects += confusion_matrix[i * vocabulary_size + i];
		for (j = 0; j < vocabulary_size; ++j)
		{
			all += confusion_matrix[i * vocabulary_size + j];
		}
	}
	if (print_confusion_matrix != 0)
	{
		printf(" ");
		for (i = 0; i < vocabulary_size; ++i)
		{
			printf("\t\t%s", references_of_words[i].wordname);
		}
		printf("\n");
		for (i = 0; i < vocabulary_size; ++i)
		{
			printf("%s", references_of_words[i].wordname);
			for (j = 0; j < vocabulary_size; ++j)
			{
				printf("\t\t%d", confusion_matrix[i * vocabulary_size + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	free(confusion_matrix);
	finalize_interesting_words(interesting_words, vocabulary_size);
	free(dp_matrix);
	free(similarities);
	free(best_similarities);
	if (all <= 0)
	{
		fprintf(stderr, "There are no interesting words in data for testing!\n");
		return 0;
	}
	accuracy = 100.0 * ((float)corrects) / ((float)all);
	printf("Total accuracy is %5.2f.\n", accuracy);
	return 1;
}

float* create_references_for_silences(TTrainDataForWord train_data, int feature_vector_size)
{
	int i, start_pos = 0, max_spectrogram_size = 0;
	float *tmp_dist_matrix;
	float *res = (float*)malloc(feature_vector_size * train_data.n * sizeof(float));
	if (res == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		return NULL;
	}
	for (i = 0; i < train_data.n; ++i)
	{
		if (train_data.spectrograms[i].n > max_spectrogram_size)
		{
			max_spectrogram_size = train_data.spectrograms[i].n;
		}
	}
	tmp_dist_matrix = (float*)malloc(max_spectrogram_size * max_spectrogram_size * sizeof(float));
	if (tmp_dist_matrix == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		free(res);
		return NULL;
	}
	for (i = 0; i < train_data.n; ++i)
	{
		find_reference_spectrum(train_data.spectrograms[i].spectrogram, train_data.spectrograms[i].n,
			feature_vector_size, &res[start_pos], tmp_dist_matrix);
		start_pos += feature_vector_size;
		printf("Reference spectrum for sound %d from %d has been successfully calculated...\n", i + 1, train_data.n);
	}
	free(tmp_dist_matrix);
	return res;
}

TReference* create_references_for_words(TTrainDataForWord train_data[],
	int speech_segments_number_for_words[], int vocabulary_size,
	int feature_vector_size, float silence_spectrums[], int number_of_silences,
	int restarts_number)
{
	int ok = 1, n_seg = 0;
	int seg_start_pos;
	int word_index, sound_index, state_index, restart_index;
	int max_spectrogram_length = 0, max_number_of_states = 0;
	float *dp_matrix, *tmp_spectrum, *tmp_dist_matrix;
	int *dp_matrix_for_lengths, *number_of_states;
	int *old_segmentation, *new_segmentation;
	float quality;
	int segmentation_diff;
	TReference *res = NULL;
	number_of_states = (int*)malloc(vocabulary_size * sizeof(int));
	if (number_of_states == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		return NULL;
	}
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		if (speech_segments_number_for_words == NULL)
		{
			number_of_states[word_index] = calculate_states_number_for_word(
				train_data[word_index].wordname);
		}
		else
		{
			number_of_states[word_index] = speech_segments_number_for_words[word_index] + 2;
		}
		if (number_of_states[word_index] > max_number_of_states)
		{
			max_number_of_states = number_of_states[word_index];
		}
		n_seg += train_data[word_index].n * number_of_states[word_index];
        for (sound_index = 0; sound_index < train_data[word_index].n; ++sound_index)
		{
			if (train_data[word_index].spectrograms[sound_index].n > max_spectrogram_length)
			{
				max_spectrogram_length = train_data[word_index].spectrograms[sound_index].n;
			}
		}
	}
	old_segmentation = (int*)malloc(n_seg * sizeof(int));
	if ((old_segmentation == NULL))
	{
		fprintf(stderr, "Out of memory!\n");
		free(number_of_states);
		return NULL;
	}
	new_segmentation = (int*)malloc(n_seg * sizeof(int));
	if ((new_segmentation == NULL))
	{
		fprintf(stderr, "Out of memory!\n");
		free(number_of_states);
		free(old_segmentation);
		return NULL;
	}
	if ((max_spectrogram_length == 0) || (max_number_of_states <= 2))
	{
		fprintf(stderr, "Spectrograms for training are wrong!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		return NULL;
	}
	tmp_dist_matrix = (float*)malloc(max_spectrogram_length * max_spectrogram_length * sizeof(float));
	if (tmp_dist_matrix == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		return NULL;
	}
	res = (TReference*)malloc(vocabulary_size * sizeof(TReference));
	if (res == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		free(tmp_dist_matrix);
		return NULL;
	}
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		res[word_index].n = 0;
		res[word_index].wordname = NULL;
		res[word_index].reference = NULL;
	}
	dp_matrix = (float*)malloc(max_spectrogram_length * max_number_of_states * sizeof(float));
	if (dp_matrix == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		free(tmp_dist_matrix);
		finalize_references(res, vocabulary_size);
		return NULL;
	}
	dp_matrix_for_lengths = (int*)malloc(max_spectrogram_length * max_number_of_states * 
		sizeof(int));
	if (dp_matrix_for_lengths == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		finalize_references(res, vocabulary_size);
		free(dp_matrix);
		free(tmp_dist_matrix);
		return NULL;
	}
	tmp_spectrum = (float*)malloc(feature_vector_size * sizeof(float));
	if (tmp_spectrum == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		finalize_references(res, vocabulary_size);
		free(dp_matrix);
		free(dp_matrix_for_lengths);
		free(tmp_dist_matrix);
		return NULL;
	}
	quality = 0.0;
	seg_start_pos = 0;
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		res[word_index].n = number_of_states[word_index] - 2;
		res[word_index].reference = (TReferenceItem*)malloc(sizeof(TReferenceItem) * 
			res[word_index].n);
		if (res[word_index].reference == NULL)
		{
			ok = 0;
			break;
		}
		for (state_index = 0; state_index < res[word_index].n; ++state_index)
		{
			res[word_index].reference[state_index].spectrum = NULL;
		}
		res[word_index].wordname = (char*)malloc(sizeof(char) * 
			(strlen(train_data[word_index].wordname) + 1));
		if (res[word_index].wordname == NULL)
		{
			ok = 0;
			break;
		}
		memset(res[word_index].wordname, 0, strlen(train_data[word_index].wordname) + 1);
		strcpy(res[word_index].wordname, train_data[word_index].wordname);
		for (state_index = 0; state_index < res[word_index].n; ++state_index)
		{
			res[word_index].reference[state_index].spectrum = (float*)malloc(feature_vector_size * 
				sizeof(float));
			if (res[word_index].reference[state_index].spectrum == NULL)
			{
				ok = 0;
				break;
			}
		}
		if (!ok)
		{
			break;
		}
		for (sound_index = 0; sound_index < train_data[word_index].n; ++sound_index)
		{
			quality += do_self_segmentation(
				train_data[word_index].spectrograms[sound_index].spectrogram,
				train_data[word_index].spectrograms[sound_index].n, feature_vector_size,
				silence_spectrums, number_of_silences, number_of_states[word_index] - 2,
				&old_segmentation[seg_start_pos + sound_index * number_of_states[word_index]],
				dp_matrix, dp_matrix_for_lengths, tmp_dist_matrix);
		}
		seg_start_pos += train_data[word_index].n * number_of_states[word_index];
	}
	if (!ok)
	{
		fprintf(stderr, "Out of memory!\n");
		free(old_segmentation);
		free(new_segmentation);
		free(number_of_states);
		finalize_references(res, vocabulary_size);
		free(dp_matrix);
		free(dp_matrix_for_lengths);
		free(tmp_spectrum);
		free(tmp_dist_matrix);
		return NULL;
	}
	select_best_references_for_words(train_data, vocabulary_size, feature_vector_size, old_segmentation,
		tmp_spectrum, res, tmp_dist_matrix);
	printf("Quality of self-segmentation is %f.\n\n", quality);
	for (restart_index = 1; restart_index <= restarts_number; ++restart_index)
	{
		printf("Restart %d:\n", restart_index);
		quality = 0.0;
		seg_start_pos = 0;
		for (word_index = 0; word_index < vocabulary_size; ++word_index)
		{
			for (sound_index = 0; sound_index < train_data[word_index].n; ++sound_index)
			{
				quality += do_segmentation(
					train_data[word_index].spectrograms[sound_index].spectrogram,
					train_data[word_index].spectrograms[sound_index].n, feature_vector_size,
					silence_spectrums, number_of_silences, res[word_index],
					&new_segmentation[seg_start_pos + sound_index * number_of_states[word_index]],
					dp_matrix, dp_matrix_for_lengths);
			}
			seg_start_pos += train_data[word_index].n * number_of_states[word_index];
		}
		select_best_references_for_words(train_data, vocabulary_size, feature_vector_size, new_segmentation,
			tmp_spectrum, res, tmp_dist_matrix);
		printf("  - quality of segmentation is %f;\n", quality);
		segmentation_diff = compare_segmentation(old_segmentation, new_segmentation,
			train_data, speech_segments_number_for_words, vocabulary_size);
		printf("  - segmentations diff is %d.\n\n", segmentation_diff);
		if (segmentation_diff <= 1)
		{
			printf("Segmentation is stabilized.\n");
			break;
		}
		memcpy(old_segmentation, new_segmentation, n_seg * sizeof(int));
	}
	find_optimal_bounds_of_references(train_data, vocabulary_size, old_segmentation, res);
	free(new_segmentation);
	free(old_segmentation);
	free(dp_matrix);
	free(dp_matrix_for_lengths);
	free(number_of_states);
	free(tmp_spectrum);
	free(tmp_dist_matrix);
	return res;
}

void select_best_references_for_words(TTrainDataForWord train_data[],
	int vocabulary_size, int feature_vector_size, int segmentation[], float tmp_reference_spectrum[],
	TReference references_vocabulary[], float tmp_dist_matrix[])
{
	int seg_start_pos;
	int word_index, sound_index;
	int number_of_states, state_index;
	int i, j, segment_start, segment_size, min_segment_size, max_segment_size;
	float quality, best_quality;
	seg_start_pos = 0;
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		best_quality = -FLT_MAX;
		number_of_states = references_vocabulary[word_index].n + 2;
		for (state_index = 1; state_index < (number_of_states - 1); ++state_index)
		{
			segment_start = 0;
			for (j = 0; j < state_index; ++j)
			{
				segment_start += segmentation[seg_start_pos + j];
			}
			segment_start *= feature_vector_size;
			segment_size = segmentation[seg_start_pos + state_index];
			min_segment_size = segment_size;
			max_segment_size = segment_size;
			best_quality = find_reference_spectrum(
				&train_data[word_index].spectrograms[0].spectrogram[segment_start], segment_size,
				feature_vector_size,
				references_vocabulary[word_index].reference[state_index - 1].spectrum, tmp_dist_matrix);
			for (i = 1; i < train_data[word_index].n; i++)
			{
				segment_start = 0;
				for (j = 0; j < state_index; ++j)
				{
					segment_start += segmentation[seg_start_pos + i * number_of_states + j];
				}
				segment_start *= feature_vector_size;
				segment_size = segmentation[seg_start_pos + i * number_of_states + state_index];
				if (segment_size > max_segment_size)
				{
					max_segment_size = segment_size;
				}
				if (segment_size < min_segment_size)
				{
					min_segment_size = segment_size;
				}
				best_quality += calculate_similarity(
					&train_data[word_index].spectrograms[i].spectrogram[segment_start], segment_size,
					feature_vector_size,
					references_vocabulary[word_index].reference[state_index - 1].spectrum);
			}
			for (sound_index = 1; sound_index < train_data[word_index].n; ++sound_index)
			{
				segment_start = 0;
				for (j = 0; j < state_index; ++j)
				{
					segment_start += segmentation[seg_start_pos + sound_index * number_of_states + j];
				}
				segment_start *= feature_vector_size;
				segment_size = segmentation[seg_start_pos + sound_index * number_of_states + state_index];
				quality = find_reference_spectrum(
					&train_data[word_index].spectrograms[sound_index].spectrogram[segment_start],
					segment_size, feature_vector_size, tmp_reference_spectrum, tmp_dist_matrix);
				for (i = 0; i < train_data[word_index].n; ++i)
				{
					if (i == sound_index)
					{
						continue;
					}
					segment_start = 0;
					for (j = 0; j < state_index; ++j)
					{
						segment_start += segmentation[seg_start_pos + i * number_of_states + j];
					}
					segment_start *= feature_vector_size;
					segment_size = segmentation[seg_start_pos + i * number_of_states + state_index];
					quality += calculate_similarity(
						&train_data[word_index].spectrograms[i].spectrogram[segment_start], segment_size,
						feature_vector_size, tmp_reference_spectrum);
				}
				if (quality > best_quality)
				{
					best_quality = quality;
					memcpy(&(references_vocabulary[word_index].reference[state_index - 1].spectrum[0]),
						tmp_reference_spectrum, sizeof(float) * feature_vector_size);
				}
			}
			if (min_segment_size == max_segment_size)
			{
				if (min_segment_size > 1)
				{
					min_segment_size -= 1;
				}
				max_segment_size += 1;
			}
			references_vocabulary[word_index].reference[state_index - 1].m = min_segment_size / 2;
			if (references_vocabulary[word_index].reference[state_index - 1].m < 1)
			{
				references_vocabulary[word_index].reference[state_index - 1].m = 1;
			}
			references_vocabulary[word_index].reference[state_index - 1].M = max_segment_size * 2;
		}
		seg_start_pos += train_data[word_index].n * number_of_states;
	}
}

void find_optimal_bounds_of_references(TTrainDataForWord train_data[], int vocabulary_size, int segmentation[],
	TReference references_vocabulary[])
{
	int seg_start_pos;
	int word_index, sound_index, state_index, number_of_states;
	int segment_length, max_segment_length, min_segment_length;
	seg_start_pos = 0;
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		number_of_states = references_vocabulary[word_index].n + 2;
		for (state_index = 1; state_index < (number_of_states - 1); ++state_index)
		{
			sound_index = 0;
			segment_length = segmentation[seg_start_pos + sound_index * number_of_states + state_index];
			max_segment_length = segment_length;
			min_segment_length = segment_length;
			for (sound_index = 1; sound_index < train_data[word_index].n; ++sound_index)
			{
				segment_length = segmentation[seg_start_pos + sound_index * number_of_states + state_index];
				if (segment_length > max_segment_length)
				{
					max_segment_length = segment_length;
				}
				if (segment_length < min_segment_length)
				{
					min_segment_length = segment_length;
				}
			}
			references_vocabulary[word_index].reference[state_index - 1].m = min_segment_length;
			references_vocabulary[word_index].reference[state_index - 1].M = max_segment_length;
		}
		seg_start_pos += train_data[word_index].n * number_of_states;
	}
}

int calculate_states_number_for_word(char* source_word)
{
	return strlen(source_word) * 3 + 2;
}

int compare_segmentation(int first_segmentation[], int second_segmentation[],
	TTrainDataForWord train_data[], int speech_segments_number_for_words[],
	int vocabulary_size)
{
	int word_index, sound_index, segment_index, number_of_states;
	int diff, total_diff = 0;
	int seg_pos = 0;
	for (word_index = 0; word_index < vocabulary_size; ++word_index)
	{
		if (speech_segments_number_for_words == NULL)
		{
			number_of_states = calculate_states_number_for_word(train_data[word_index].wordname);
		}
		else
		{
			number_of_states = speech_segments_number_for_words[word_index] + 2;
		}
		for (sound_index = 0; sound_index < train_data[word_index].n; ++sound_index)
		{
			for (segment_index = 0; segment_index < number_of_states; ++segment_index, ++seg_pos)
			{
				diff = abs(first_segmentation[seg_pos] - second_segmentation[seg_pos]);
				if (diff > total_diff)
				{
					total_diff = diff;
				}
			}
		}
	}
	return total_diff;
}

int load_references(char* filename, TReference** references, int* vocabulary_size,
	int* feature_vector_size, float** silence_spectrums, int* silences_number)
{
	size_t nread;
	int i, j, k, states_number, m, M;
	int file_size, length, it;
	int ind1, ind2, ind3, ind4, ind5;
	struct stat filestatus;
	FILE* fp;
	char* file_contents;
	json_char* json;
	json_value *value, *silences_value, *words_value, *cur_word_value, *reference_item_value;
	*references = NULL;
	*silence_spectrums = NULL;
	*vocabulary_size = 0;
	*feature_vector_size = 0;
	*silences_number = 0;
	if (stat(filename, &filestatus) != 0)
	{
		fprintf(stderr, "File `%s` does not exist!\n", filename);
		return 0;
	}
	file_size = filestatus.st_size;
	file_contents = (char*)malloc((file_size + 1) * sizeof(char));
	if (file_contents == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		return 0;
	}
	memset(file_contents, 0, (file_size + 1) * sizeof(char));
	fp = fopen(filename, "r");
	if (fp == NULL)
	{
		fprintf(stderr, "File `%s` cannot be opened for reading!\n%s", filename, strerror(errno));
		free(file_contents);
		return 0;
	}
	nread = fread(file_contents, 1, file_size * sizeof(char), fp);
	if (nread < 1)
	{
		fprintf(stderr, "Data cannot be read from the file `%s`!\n", filename);
		fclose(fp);
		free(file_contents);
		return 0;
	}
	fclose(fp);
	json = (json_char*)file_contents;
	value = json_parse(json, file_size);
	if (value->type != json_object)
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	length = value->u.object.length;
	ind1 = -1; ind2 = -1; ind3 = -1; ind4 = -1; ind5 = -1;
	for (it = 0; it < length; ++it)
	{
		if (strcmp(value->u.object.values[it].name, "feature_vector_size") == 0)
		{
			ind1 = it;
		}
		else if (strcmp(value->u.object.values[it].name, "silences_number") == 0)
		{
			ind2 = it;
		}
		else if (strcmp(value->u.object.values[it].name, "silences") == 0)
		{
			ind3 = it;
		}
		else if (strcmp(value->u.object.values[it].name, "vocabulary_size") == 0)
		{
			ind4 = it;
		}
		else if (strcmp(value->u.object.values[it].name, "words") == 0)
		{
			ind5 = it;
		}
	}
	if ((ind1 < 0) || (ind2 < 0) || (ind3 < 0) || (ind4 < 0) || (ind5 < 0))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	if ((value->u.object.values[ind1].value->type != json_integer) || (value->u.object.values[ind2].value->type != json_integer) ||
		(value->u.object.values[ind3].value->type != json_array) || (value->u.object.values[ind4].value->type != json_integer) ||
		(value->u.object.values[ind5].value->type != json_array))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	*feature_vector_size = (int)value->u.object.values[ind1].value->u.integer;
	*vocabulary_size = (int)value->u.object.values[ind4].value->u.integer;
	*silences_number = (int)value->u.object.values[ind2].value->u.integer;
	if ((*feature_vector_size <= 0) || (*silences_number <= 0) || (*vocabulary_size <= 0))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		*silences_number = 0;
		return 0;
	}
	silences_value = value->u.object.values[ind3].value;
	words_value = value->u.object.values[ind5].value;
	if ((silences_value->u.array.length != *silences_number) ||
		(words_value->u.array.length != *vocabulary_size))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		*silences_number = 0;
		return 0;
	}
	int ok = 1;
	*silence_spectrums = (float*)malloc(*feature_vector_size * *silences_number * sizeof(float));
	if (*silence_spectrums == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		json_value_free(value);
		free(file_contents);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		*silences_number = 0;
		return 0;
	}
	for (i = 0; i < *silences_number; ++i)
	{
		if (silences_value->u.array.values[i]->type != json_array)
		{
			ok = 0;
			break;
		}
		if (silences_value->u.array.values[i]->u.array.length != *feature_vector_size)
		{
			ok = 0;
			break;
		}
		for (j = 0; j < *feature_vector_size; ++j)
		{
			if (silences_value->u.array.values[i]->u.array.values[j]->type != json_double)
			{
				ok = 0;
				break;
			}
			(*silence_spectrums)[i * *feature_vector_size + j] = 
				silences_value->u.array.values[i]->u.array.values[j]->u.dbl;
		}
		if (!ok)
		{
			break;
		}
	}
	if (!ok)
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		free(*silence_spectrums);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		*silences_number = 0;
		*silence_spectrums = NULL;
		return 0;
	}
	*references = (TReference*)malloc(*vocabulary_size * sizeof(TReference));
	if (*references == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		json_value_free(value);
		free(file_contents);
		free(*silence_spectrums);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		*silences_number = 0;
		*silence_spectrums = NULL;
		return 0;
	}
	for (i = 0; i < *vocabulary_size; ++i)
	{
		(*references)[i].n = 0;
		(*references)[i].reference = NULL;
		(*references)[i].wordname = NULL;
	}
	for (i = 0; i < *vocabulary_size; ++i)
	{
		if (words_value->u.array.values[i]->type != json_object)
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		cur_word_value = words_value->u.array.values[i];
		ind1 = -1; ind2 = -1; ind3 = -1;
		for (it = 0; it < cur_word_value->u.object.length; ++it)
		{
			if (strcmp(cur_word_value->u.object.values[it].name, "n") == 0)
			{
				ind1 = it;
			}
			if (strcmp(cur_word_value->u.object.values[it].name, "wordname") == 0)
			{
				ind2 = it;
			}
			if (strcmp(cur_word_value->u.object.values[it].name, "reference") == 0)
			{
				ind3 = it;
			}
		}
		if ((ind1 < 0) || (ind2 < 0) || (ind3 < 0))
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		if ((cur_word_value->u.object.values[ind1].value->type != json_integer) ||
			(cur_word_value->u.object.values[ind2].value->type != json_string) ||
			(cur_word_value->u.object.values[ind3].value->type != json_array))
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		states_number = (int)cur_word_value->u.object.values[ind1].value->u.integer;
		if (states_number <= 0)
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		if (states_number != cur_word_value->u.object.values[ind3].value->u.array.length)
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		length = cur_word_value->u.object.values[ind2].value->u.string.length;
		(*references)[i].wordname = (char*)malloc((length + 1) * sizeof(char));
		if ((*references)[i].wordname == NULL)
		{
			ok = 0;
			fprintf(stderr, "Out of memory!\n");
			break;
		}
		memset((*references)[i].wordname, 0, (length + 1) * sizeof(char));
		strcpy((*references)[i].wordname, cur_word_value->u.object.values[ind2].value->u.string.ptr);
		(*references)[i].reference = (TReferenceItem*)malloc(states_number * sizeof(TReferenceItem));
		if ((*references)[i].reference == NULL)
		{
			ok = 0;
			fprintf(stderr, "Out of memory!\n");
			break;
		}
		(*references)[i].n = states_number;
		for (j = 0; j < states_number; ++j)
		{
			(*references)[i].reference[j].m = 0;
			(*references)[i].reference[j].M = 0;
			(*references)[i].reference[j].spectrum = NULL;
		}
		for (j = 0; j < states_number; ++j)
		{
			reference_item_value = cur_word_value->u.object.values[ind3].value->u.array.values[j];
			if (reference_item_value->type != json_object)
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			ind1 = -1; ind2 = -1; ind3 = -1;
			for (it = 0; it < reference_item_value->u.object.length; ++it)
			{
				if (strcmp(reference_item_value->u.object.values[it].name, "m") == 0)
				{
					ind1 = it;
				}
				if (strcmp(reference_item_value->u.object.values[it].name, "M") == 0)
				{
					ind2 = it;
				}
				if (strcmp(reference_item_value->u.object.values[it].name, "spectrum") == 0)
				{
					ind3 = it;
				}
			}
			if ((ind1 < 0) || (ind2 < 0) || (ind3 < 0))
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			if ((reference_item_value->u.object.values[ind1].value->type != json_integer) ||
				(reference_item_value->u.object.values[ind2].value->type != json_integer) ||
				(reference_item_value->u.object.values[ind3].value->type != json_array))
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			m = (int)reference_item_value->u.object.values[ind1].value->u.integer;
			M = (int)reference_item_value->u.object.values[ind2].value->u.integer;
			if ((m < 1) || (M < m))
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			if (reference_item_value->u.object.values[ind3].value->u.array.length != *feature_vector_size)
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			(*references)[i].reference[j].m = m;
			(*references)[i].reference[j].M = M;
			(*references)[i].reference[j].spectrum = (float*)malloc(*feature_vector_size * sizeof(float));
			if ((*references)[i].reference[j].spectrum == NULL)
			{
				ok = 0;
				fprintf(stderr, "Out of memory!\n");
				break;
			}
			for (k = 0; k < *feature_vector_size; ++k)
			{
				if (reference_item_value->u.object.values[ind3].value->u.array.values[k]->type != json_double)
				{
					ok = 0;
					fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
					break;
				}
				(*references)[i].reference[j].spectrum[k] = 
					reference_item_value->u.object.values[ind3].value->u.array.values[k]->u.dbl;
			}
			if (!ok)
			{
				break;
			}
		}
		if (!ok)
		{
			break;
		}
	}
	json_value_free(value);
	free(file_contents);
	if (!ok)
	{
		finalize_references(*references, *vocabulary_size);
		free(*silence_spectrums);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		*silences_number = 0;
		*silence_spectrums = NULL;
		*references = NULL;
		return 0;
	}
	return 1;
}

int save_references(char* filename, TReference references[], int vocabulary_size,
	int feature_vector_size, float silence_spectrums[], int silences_number)
{
	int i, j, k;
	FILE* fp = fopen(filename, "w");
	if (fp == NULL)
	{
		fprintf(stderr, "File `%s` cannot be opened for writing!\n%s\n", filename, strerror(errno));
		return 0;
	}
	fprintf(fp, "{\n");
	fprintf(fp, "    \"feature_vector_size\": %d,\n", feature_vector_size);
	fprintf(fp, "    \"silences_number\": %d,\n", silences_number);
	fprintf(fp, "    \"silences\": [\n");
	for (i = 0; i < silences_number; ++i)
	{
		fprintf(fp, "        [%.12f", silence_spectrums[i * feature_vector_size]);
		for (j = 1; j < feature_vector_size; ++j)
		{
			fprintf(fp, ", %.12f", silence_spectrums[i * feature_vector_size + j]);
		}
		if (i < (silences_number - 1))
		{
			fprintf(fp, "],\n");
		}
		else
		{
			fprintf(fp, "]\n");
		}
	}
	fprintf(fp, "    ],\n");
	fprintf(fp, "    \"vocabulary_size\": %d,\n", vocabulary_size);
	fprintf(fp, "    \"words\": [\n");
	for (i = 0; i < vocabulary_size; ++i)
	{
		fprintf(fp, "        {\n");
		fprintf(fp, "            \"wordname\": \"%s\",\n", references[i].wordname);
		fprintf(fp, "            \"n\": %d,\n", references[i].n);
		fprintf(fp, "            \"reference\": [\n");
		for (j = 0; j < references[i].n; ++j)
		{
			fprintf(fp, "                {\n");
			fprintf(fp, "                    \"m\": %d,\n", references[i].reference[j].m);
			fprintf(fp, "                    \"M\": %d,\n", references[i].reference[j].M);
			fprintf(fp, "                    \"spectrum\": [%.12f",
				references[i].reference[j].spectrum[0]);
			for (k = 1; k < feature_vector_size; ++k)
			{
				fprintf(fp, ", %.12f", references[i].reference[j].spectrum[k]);
			}
			fprintf(fp, "]\n");
			if (j < (references[i].n - 1))
			{
				fprintf(fp, "                },\n");
			}
			else
			{
				fprintf(fp, "                }\n");
			}
		}
		fprintf(fp, "            ]\n");
		if (i < (vocabulary_size - 1))
		{
			fprintf(fp, "        },\n");
		}
		else
		{
			fprintf(fp, "        }\n");
		}
	}
	fprintf(fp, "    ]\n");
	fprintf(fp, "}\n");
	fclose(fp);
	return 1;
}

void finalize_references(TReference* references, int vocabulary_size)
{
	int i, j;
	for (i = 0; i < vocabulary_size; ++i)
	{
		if (references[i].wordname != NULL)
		{
			free(references[i].wordname);
			references[i].wordname = NULL;
		}
		if (references[i].reference != NULL)
		{
			for (j = 0; j < references[i].n; ++j)
			{
				if (references[i].reference[j].spectrum != NULL)
				{
					free(references[i].reference[j].spectrum);
					references[i].reference[j].spectrum = NULL;
				}
			}
			free(references[i].reference);
			references[i].reference = NULL;
		}
	}
	free(references);
}

void finalize_train_data(TTrainDataForWord* train_data, int vocabulary_size)
{
	int i;
	for (i = 0; i < vocabulary_size; ++i)
	{
		finalize_train_data_for_word(train_data[i]);
	}
	free(train_data);
}

void finalize_train_data_for_word(TTrainDataForWord train_data_for_word)
{
	int j;
	if (train_data_for_word.wordname != NULL)
	{
		free(train_data_for_word.wordname);
		train_data_for_word.wordname = NULL;
	}
	if (train_data_for_word.spectrograms != NULL)
	{
		for (j = 0; j < train_data_for_word.n; ++j)
		{
			if (train_data_for_word.spectrograms[j].spectrogram != NULL)
			{
				free(train_data_for_word.spectrograms[j].spectrogram);
				train_data_for_word.spectrograms[j].spectrogram = NULL;
			}
		}
		free(train_data_for_word.spectrograms);
		train_data_for_word.spectrograms = NULL;
	}
}

void finalize_spectrograms_list(TSpectrogram* spectrograms, int spectrograms_number)
{
	int i;
	for (i = 0; i < spectrograms_number; ++i)
	{
		if (spectrograms[i].spectrogram != NULL)
		{
			free(spectrograms[i].spectrogram);
			spectrograms[i].spectrogram = NULL;
		}
	}
	free(spectrograms);
}

char* join_and_prepare_filename(char *basedir, char* filename)
{
	char c;
	int i;
	int n1 = strlen(basedir);
	int n2 = strlen(filename);
	if (n1 > 0)
	{
		if ((basedir[n1 - 1] == '/') || (basedir[n1 - 1] == '\\'))
		{
			memmove(&filename[n1], &filename[0], (n2 + 1) * sizeof(char));
			memcpy(&filename[0], basedir, n1);
		}
		else
		{
			i = 0;
			while (i < n1)
			{
				if ((basedir[i] == '/') || (basedir[i] == '\\'))
				{
					c = basedir[i];
					break;
				}
				++i;
			}
			if (i >= n1)
			{
				i = 0;
				while (i < n2)
				{
					if ((filename[i] == '/') || (filename[i] == '\\'))
					{
						c = filename[i];
						break;
					}
					++i;
				}
				if (i >= n2)
				{
					c = '/';
				}
			}
			memmove(&filename[n1 + 1], &filename[0], (n2 + 1) * sizeof(char));
			memcpy(&filename[0], basedir, n1);
			filename[n1] = c;
		}
	}
	i = 0;
	while (filename[i] != 0)
	{
		if ((filename[i] == '/') || (filename[i] == '\\'))
		{
			filename[i] = DIRCHAR;
		}
		++i;
	}
	filename[i] = '.';
	filename[i + 1] = 'b';
	filename[i + 2] = 'i';
	filename[i + 3] = 'n';
	filename[i + 4] = 0;
	return filename;
}

int load_spectrogram(char* filename, float** spectrogram,
	int* spectrogram_size, int* feature_vector_size)
{
	int i, ok = 1;
	int32_t a, b;
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		fprintf(stderr, "File `%s` cannot be opened for reading!\n%s\n", filename, strerror(errno));
		return 0;
	}
	if (fread(&a, sizeof(int32_t), 1, fp) != 1)
	{
		fprintf(stderr, "File `%s` is wrong!\n", filename);
		fclose(fp);
		return 0;
	}
	if (fread(&b, sizeof(int32_t), 1, fp) != 1)
	{
		fprintf(stderr, "File `%s` is wrong!\n", filename);
		fclose(fp);
		return 0;
	}
	if ((a <= 0) || (b <= 0))
	{
		fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", filename);
		fclose(fp);
		return 0;
	}
	*spectrogram_size = a;
	*feature_vector_size = b;
	*spectrogram = (float*)malloc(*spectrogram_size * *feature_vector_size * sizeof(float));
	if (*spectrogram == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		fclose(fp);
		return 0;
	}
	if (fread(*spectrogram, sizeof(float), *spectrogram_size * *feature_vector_size, fp) != 
		(*spectrogram_size * *feature_vector_size))
	{
		fprintf(stderr, "File `%s` is wrong!\n", filename);
		free(*spectrogram);
		fclose(fp);
		return 0;
	}
	for (i = 0; i < (*spectrogram_size * *feature_vector_size); ++i)
	{
		if ((*spectrogram)[i] < 0.0)
		{
			ok = 0;
			break;
		}
	}
	if (!ok)
	{
		fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", filename);
		free(*spectrogram);
		fclose(fp);
		return 0;
	}
	fclose(fp);
	return 1;
}

int load_list_of_spectrograms(char* filename, char* basedir,
	TSpectrogram** spectrograms, int* spectrograms_number, int* feature_vector_size)
{
	int i, n = 0;
	int ft_size = -1;
	int spectrogram_sizes[2];
	char buffer[BUFFER_SIZE];
	FILE *fp = fopen(filename, "r");
	if (fp == NULL)
	{
		fprintf(stderr, "File `%s` cannot be opened for reading!\n%s\n", filename, strerror(errno));
		return 0;
	}
	memset(buffer, 0, BUFFER_SIZE);
	while (fgets(buffer, BUFFER_SIZE - 1, fp) != NULL)
	{
		if (strlen(buffer) > 0)
		{
			n += 1;
		}
		memset(buffer, 0, BUFFER_SIZE);
		if (feof(fp))
		{
			break;
		}
	}
	if (n == 0)
	{
		fprintf(stderr, "File `%s` is empty!\n", filename);
		fclose(fp);
		return 0;
	}
	rewind(fp);
	*spectrograms_number = n;
	*spectrograms = (TSpectrogram*)malloc(n * sizeof(TSpectrogram));
	if (*spectrograms == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		fclose(fp);
		return 0;
	}
	for (i = 0; i < n; ++i)
	{
		(*spectrograms)[i].n = 0;
		(*spectrograms)[i].spectrogram = NULL;
	}
	i = 0;
	while (n > 0)
	{
		if (feof(fp))
		{
			fprintf(stderr, "List of spectrograms from the file `%s` is wrong!\n", filename);
			break;
		}
		memset(buffer, 0, BUFFER_SIZE);
		if (fgets(buffer, BUFFER_SIZE - 1, fp) == NULL)
		{
			fprintf(stderr, "List of spectrograms from the file `%s` is wrong!\n", filename);
			break;
		}
		if (strlen(buffer) > 0)
		{
			if (!load_spectrogram(join_and_prepare_filename(basedir, buffer),
				&((*spectrograms)[i].spectrogram), &spectrogram_sizes[0], &spectrogram_sizes[1]))
			{
				break;
			}
			if ((spectrogram_sizes[0] <= 0) || (spectrogram_sizes[1] <= 0))
			{
				fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", buffer);
				break;
			}
			if (ft_size < 0)
			{
				ft_size = spectrogram_sizes[1];
			}
			else
			{
				if (ft_size != spectrogram_sizes[1])
				{
					fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", buffer);
					break;
				}
			}
			(*spectrograms)[i].n = spectrogram_sizes[0];
			n -= 1;
			i += 1;
		}
	}
	fclose(fp);
	if (n > 0)
	{
		finalize_spectrograms_list(*spectrograms, *spectrograms_number);
		return 0;
	}
	*feature_vector_size = ft_size;
	return 1;
}

int load_train_data(char* filename, char* basedir, char* datapart,
	char** interesting_words, int number_of_interesting_words,
	int* feature_vector_size, TTrainDataForWord** train_data_for_words, int* vocabulary_size,
	TTrainDataForWord* train_data_for_silences)
{
	int ok = 1;
	size_t nread;
	int n, i, j, ft_size = -1;
	int spectrogram_sizes[2];
	char buffer[BUFFER_SIZE];
	int file_size, length, it, it2;
	int ind1, ind2, ind3;
	struct stat filestatus;
	FILE* fp;
	char* file_contents;
	json_char* json;
	json_value *value, *part_value, *word_value, *sound_value, *filename_value;
	*train_data_for_words = NULL;
	*vocabulary_size = 0;
	*feature_vector_size = 0;
	(*train_data_for_silences).wordname = NULL;
	(*train_data_for_silences).spectrograms = NULL;
	(*train_data_for_silences).n = 0;

	if ((strcmp(datapart, "train") != 0) && (strcmp(datapart, "test") != 0) && 
		(strcmp(datapart, "validation") != 0))
	{
		fprintf(stderr, "`%s` is unknown data part!\n", datapart);
		return 0;
	}
	if (stat(filename, &filestatus) != 0)
	{
		fprintf(stderr, "File `%s` does not exist!\n", filename);
		return 0;
	}
	file_size = filestatus.st_size;
	file_contents = (char*)malloc((file_size + 1) * sizeof(char));
	if (file_contents == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		return 0;
	}
	memset(file_contents, 0, (file_size + 1) * sizeof(char));
	fp = fopen(filename, "r");
	if (fp == NULL)
	{
		fprintf(stderr, "File `%s` cannot be opened for reading!\n%s\n", filename, strerror(errno));
		free(file_contents);
		return 0;
	}
	nread = fread(file_contents, 1, file_size, fp);
	if (nread < 1)
	{
		fprintf(stderr, "Data cannot be read from the file `%s`!\n", filename);
		fclose(fp);
		free(file_contents);
		return 0;
	}
	fclose(fp);
	json = (json_char*)file_contents;
	value = json_parse(json, file_size);
	if (value->type != json_object)
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	length = value->u.object.length;
	ind1 = -1; ind2 = -1; ind3 = -1;
	for (it = 0; it < length; ++it)
	{
		if (strcmp(value->u.object.values[it].name, "train") == 0)
		{
			ind1 = it;
		}
		else if (strcmp(value->u.object.values[it].name, "test") == 0)
		{
			ind2 = it;
		}
		else if (strcmp(value->u.object.values[it].name, "validation") == 0)
		{
			ind3 = it;
		}
	}
	if ((ind1 < 0) || (ind2 < 0) || (ind3 < 0))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	if (strcmp(datapart, "train") == 0)
	{
		part_value = value->u.object.values[ind1].value;
	}
	else if (strcmp(datapart, "test") == 0)
	{
		part_value = value->u.object.values[ind2].value;
	}
	else
	{
		part_value = value->u.object.values[ind3].value;
	}
	if (part_value->type != json_object)
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	length = part_value->u.object.length;
	ind1 = -1; ind2 = -1;
	for (it = 0; it < length; ++it)
	{
		if (strcmp(part_value->u.object.values[it].name, "speech") == 0)
		{
			ind1 = it;
		}
		else if (strcmp(part_value->u.object.values[it].name, "silence") == 0)
		{
			ind2 = it;
		}
	}
	if ((ind1 < 0) || (ind2 < 0))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	if ((part_value->u.object.values[ind1].value->type != json_object) ||
		(part_value->u.object.values[ind2].value->type != json_array))
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	if (part_value->u.object.values[ind2].value->u.array.length <= 0)
	{
		fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	train_data_for_silences->n = part_value->u.object.values[ind2].value->u.array.length;
	train_data_for_silences->spectrograms = (TSpectrogram*)malloc(sizeof(TSpectrogram) * 
		train_data_for_silences->n);
	if (train_data_for_silences->spectrograms == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		json_value_free(value);
		free(file_contents);
		train_data_for_silences->n = 0;
		return 0;
	}
	for (i = 0; i < train_data_for_silences->n; ++i)
	{
		train_data_for_silences->spectrograms[i].n = 0;
		train_data_for_silences->spectrograms[i].spectrogram = NULL;
	}
	for (i = 0; i < train_data_for_silences->n; ++i)
	{
		filename_value = part_value->u.object.values[ind2].value->u.array.values[i];
		if (filename_value->type != json_string)
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		memset(buffer, 0, BUFFER_SIZE);
		strcpy(buffer, filename_value->u.string.ptr);
		if (!load_spectrogram(join_and_prepare_filename(basedir, buffer),
			&(train_data_for_silences->spectrograms[i].spectrogram),
			&spectrogram_sizes[0], &spectrogram_sizes[1]))
		{
			ok = 0;
			break;
		}
		if ((spectrogram_sizes[0] <= 0) || (spectrogram_sizes[1] <= 0))
		{
			ok = 0;
			fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", buffer);
			break;
		}
		if (ft_size < 0)
		{
			ft_size = spectrogram_sizes[1];
		}
		else
		{
			if (ft_size != spectrogram_sizes[1])
			{
				ok = 0;
				fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", buffer);
				break;
			}
		}
		train_data_for_silences->spectrograms[i].n = spectrogram_sizes[0];
	}
	if (!ok)
	{
		finalize_train_data_for_word(*train_data_for_silences);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	n = 0;
	for (it = 0; it < part_value->u.object.values[ind1].value->u.object.length; ++it)
	{
		if ((interesting_words == NULL) || (number_of_interesting_words <= 0))
		{
			++n;
		}
		else
		{
			if (find_word(part_value->u.object.values[ind1].value->u.object.values[it].name,
				interesting_words, number_of_interesting_words) >= 0)
			{
				++n;
			}
		}
	}
	if (n <= 0)
	{
		fprintf(stderr, "There are no interesting words in data for training!\n");
		finalize_train_data_for_word(*train_data_for_silences);
		json_value_free(value);
		free(file_contents);
		return 0;
	}
	*feature_vector_size = ft_size;
	*vocabulary_size = n;
	*train_data_for_words = (TTrainDataForWord*)malloc(n * sizeof(TTrainDataForWord));
	if (*train_data_for_words == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		finalize_train_data_for_word(*train_data_for_silences);
		json_value_free(value);
		free(file_contents);
		*feature_vector_size = 0;
		*vocabulary_size = 0;
		return 0;
	}
	for (i = 0; i < n; ++i)
	{
		(*train_data_for_words)[i].wordname = NULL;
		(*train_data_for_words)[i].spectrograms = NULL;
		(*train_data_for_words)[i].n = 0;
	}
	i = 0;
	for (it = 0; it < part_value->u.object.values[ind1].value->u.object.length; ++it)
	{
		if ((interesting_words != NULL) && (number_of_interesting_words > 0))
		{
			if (find_word(part_value->u.object.values[ind1].value->u.object.values[it].name,
				interesting_words, number_of_interesting_words) < 0)
			{
				continue;
			}
		}
		word_value = part_value->u.object.values[ind1].value->u.object.values[it].value;
		if (word_value->type != json_array)
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		if (word_value->u.array.length <= 0)
		{
			ok = 0;
			fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
			break;
		}
		(*train_data_for_words)[i].wordname = (char*)malloc(sizeof(char) *
			(part_value->u.object.values[ind1].value->u.object.values[it].name_length + 1));
		if ((*train_data_for_words)[i].wordname == NULL)
		{
			ok = 0;
			fprintf(stderr, "Out of memory!\n");
			break;
		}
		memset((*train_data_for_words)[i].wordname, 0, sizeof(char) *
			(part_value->u.object.values[ind1].value->u.object.values[it].name_length + 1));
		strcpy((*train_data_for_words)[i].wordname,
			part_value->u.object.values[ind1].value->u.object.values[it].name);
		(*train_data_for_words)[i].n = word_value->u.array.length;
		(*train_data_for_words)[i].spectrograms = (TSpectrogram*)malloc(sizeof(TSpectrogram) *
			(*train_data_for_words)[i].n);
		if ((*train_data_for_words)[i].spectrograms == NULL)
		{
			ok = 0;
			fprintf(stderr, "Out of memory!\n");
			break;
		}
		for (j = 0; j < (*train_data_for_words)[i].n; ++j)
		{
			(*train_data_for_words)[i].spectrograms[j].n = 0;
			(*train_data_for_words)[i].spectrograms[j].spectrogram = NULL;
		}
		for (j = 0; j < (*train_data_for_words)[i].n; ++j)
		{
			sound_value = word_value->u.array.values[j];
			if (sound_value->type != json_object)
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			ind2 = -1;
			for (it2 = 0; it2 < sound_value->u.object.length; ++it2)
			{
				if (strcmp(sound_value->u.object.values[it2].name, "source") == 0)
				{
					ind2 = it2;
					break;
				}
			}
			if (ind2 < 0)
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			filename_value = sound_value->u.object.values[ind2].value;
			if (filename_value->type != json_string)
			{
				ok = 0;
				fprintf(stderr, "Data from the file `%s` are wrong!\n", filename);
				break;
			}
			memset(buffer, 0, BUFFER_SIZE);
			strcpy(buffer, filename_value->u.string.ptr);
			if (!load_spectrogram(join_and_prepare_filename(basedir, buffer),
				&((*train_data_for_words)[i].spectrograms[j].spectrogram),
				&spectrogram_sizes[0], &spectrogram_sizes[1]))
			{
				ok = 0;
				break;
			}
			if ((spectrogram_sizes[0] <= 0) || (spectrogram_sizes[1] <= 0))
			{
				ok = 0;
				fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", buffer);
				break;
			}
			if (*feature_vector_size != spectrogram_sizes[1])
			{
				ok = 0;
				fprintf(stderr, "Spectrogram from the file `%s` is wrong!\n", buffer);
				break;
			}
			(*train_data_for_words)[i].spectrograms[j].n = spectrogram_sizes[0];
		}
		if (!ok)
		{
			break;
		}
		++i;
	}
	json_value_free(value);
	free(file_contents);
	if (!ok)
	{
		finalize_train_data_for_word(*train_data_for_silences);
		finalize_train_data(*train_data_for_words, *vocabulary_size);
		*train_data_for_words = NULL;
		*vocabulary_size = 0;
		*feature_vector_size = 0;
		return 0;
	}
	return 1;
}

int find_word(char* source_word, char** interesting_words, int number_of_interesting_words)
{
	int i, res = -1;
	for (i = 0; i < number_of_interesting_words; ++i)
	{
		if (strcmp(source_word, interesting_words[i]) == 0)
		{
			res = i;
			break;
		}
	}
	return res;
}

char* strip_line(char* source_line)
{
	int i, j;
	i = 0;
	while (source_line[i] != 0)
	{
		if ((source_line[i] != ' ') && (source_line[i] > 13))
		{
			break;
		}
		i++;
	}
	if (source_line[i] == 0)
	{
		source_line[0] = 0;
	}
	else
	{
		j = i + 1;
		while (source_line[j] != 0)
		{
			j++;
		}
		j--;
		while (j > i)
		{
			if ((source_line[j] != ' ') && (source_line[j] > 13))
			{
				break;
			}
			j--;
		}
		source_line[j + 1] = 0;
		memmove(&source_line[0], &source_line[i], (j - i + 2) * sizeof(char));
	}
	return source_line;
}

int load_interesting_words(char* filename, char*** interesting_words, int* number_of_interesting_words)
{
	int i, n, len;
	char buffer[BUFFER_SIZE];
	FILE *fp = fopen(filename, "r");
	if (fp == NULL)
	{
		fprintf(stderr, "File `%s` cannot be opened for reading!\n%s\n", filename, strerror(errno));
		return 0;
	}
	memset(buffer, 0, BUFFER_SIZE);
	n = 0;
	while (fgets(buffer, BUFFER_SIZE - 1, fp) != NULL)
	{
		if (strlen(strip_line(buffer)) > 0)
		{
			n += 1;
		}
		memset(buffer, 0, BUFFER_SIZE);
		if (feof(fp))
		{
			break;
		}
	}
	if (n == 0)
	{
		fprintf(stderr, "File `%s` is empty!\n", filename);
		fclose(fp);
		return 0;
	}
	rewind(fp);
	*number_of_interesting_words = n;
	*interesting_words = (char**)malloc(n * sizeof(char*));
	if (*interesting_words == NULL)
	{
		fprintf(stderr, "Out of memory!\n");
		fclose(fp);
		return 0;
	}
	for (i = 0; i < n; ++i)
	{
		(*interesting_words)[i] = NULL;
	}
	i = 0;
	while (n > 0)
	{
		if (feof(fp))
		{
			fprintf(stderr, "List of interesting words from the file `%s` is wrong!\n", filename);
			break;
		}
		memset(buffer, 0, BUFFER_SIZE);
		if (fgets(buffer, BUFFER_SIZE - 1, fp) == NULL)
		{
			fprintf(stderr, "List of interesting words from the file `%s` is wrong!\n", filename);
			break;
		}
		strip_line(buffer);
		len = strlen(buffer);
		if (len > 0)
		{
			(*interesting_words)[i] = (char*)malloc((len + 1) * sizeof(char));
			if ((*interesting_words)[i] == NULL)
			{
				fprintf(stderr, "Out of memory!\n");
				break;
			}
			memset((*interesting_words)[i], 0, (len + 1) * sizeof(char));
			strcpy((*interesting_words)[i], buffer);
			n -= 1;
			i += 1;
		}
	}
	if (n > 0)
	{
		finalize_interesting_words(*interesting_words, *number_of_interesting_words);
		return 0;
	}
	return 1;
}

void finalize_interesting_words(char** interesting_words, int number_of_interesting_words)
{
	int i;
	for (i = 0; i < number_of_interesting_words; ++i)
	{
		if (interesting_words[i] != NULL)
		{
			free(interesting_words[i]);
			interesting_words[i] = NULL;
		}
	}
	free(interesting_words);
}
