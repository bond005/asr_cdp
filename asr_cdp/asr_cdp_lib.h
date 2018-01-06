typedef struct _TReferenceItem {
	float *spectrum;
	int m, M;
} TReferenceItem;
typedef struct _TReference {
	char* wordname;
	TReferenceItem *reference;
	int n;
} TReference;
typedef struct _TSpectrogram {
	float *spectrogram;
	int n;
} TSpectrogram;
typedef struct _TTrainDataForWord {
	char* wordname;
	TSpectrogram *spectrograms;
	int n;
} TTrainDataForWord;

float calculate_similarity(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float reference_spectrum[]);

float find_reference_spectrum(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float reference_spectrum[]);

int recognize_one_sound(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float silence_spectrums[], int number_of_silences,
	TReference references[], int vocabulary_size,
	float best_similarities[], float similarities[], float dp_matrix[]);

int recognize_all(TSpectrogram spectrograms[], int spectrograms_number,
	int feature_vector_size, float silence_spectrums[], int number_of_silences,
	TReference references_of_words[], int vocabulary_size, char* output_file);

float do_segmentation(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float silence_spectrums[], int number_of_silences,
	TReference reference, int lengths_of_segments[],
	float dp_matrix[], int dp_matrix_for_lengths[]);

float do_self_segmentation(float spectrogram[], int spectrogram_size, int feature_vector_size,
	float silence_spectrums[], int number_of_silences,
	int speech_segments_number, int lengths_of_segments[],
	float dp_matrix[], int dp_matrix_for_lengths[]);

int evaluate(TTrainDataForWord test_data[], int all_words_number,
	int feature_vector_size, float silence_spectrums[], int number_of_silences,
	TReference references_of_words[], int vocabulary_size, int print_confusion_matrix);

float* create_references_for_silences(TTrainDataForWord train_data, int feature_vector_size);

TReference* create_references_for_words(TTrainDataForWord train_data[],
	int speech_segments_number_for_words[], int vocabulary_size,
	int feature_vector_size, float silence_spectrums[], int number_of_silences,
	int restarts_number);

void select_best_references_for_words(TTrainDataForWord train_data[],
	int vocabulary_size, int feature_vector_size, int segmentation[], float tmp_reference_spectrum[],
	TReference references_vocabulary[]);

void find_optimal_bounds_of_references(TTrainDataForWord train_data[], int vocabulary_size, int segmentation[],
	TReference references_vocabulary[]);

int calculate_states_number_for_word(char* source_word);

int compare_segmentation(int first_segmentation[], int second_segmentation[],
	TTrainDataForWord train_data[], int speech_segments_number_for_words[],
	int vocabulary_size);

int load_references(char* filename, TReference** references, int* vocabulary_size,
	int* feature_vector_size, float** silence_spectrums, int* silences_number);

int save_references(char* filename, TReference references[], int vocabulary_size,
	int feature_vector_size, float silence_spectrums[], int silences_number);

void finalize_references(TReference* references, int vocabulary_size);

void finalize_train_data(TTrainDataForWord* train_data, int vocabulary_size);

void finalize_train_data_for_word(TTrainDataForWord train_data_for_word);

void finalize_spectrograms_list(TSpectrogram* spectrograms, int spectrograms_number);

char* join_and_prepare_filename(char *basedir, char* filename);

int load_spectrogram(char* filename, float** spectrogram,
	int* spectrogram_size, int* feature_vector_size);

int load_list_of_spectrograms(char* filename, char* basedir,
	TSpectrogram** spectrograms, int* spectrograms_number, int* feature_vector_size);

int load_train_data(char* filename, char* basedir, char* datapart,
	char** interesting_words, int number_of_interesting_words,
	int* feature_vector_size, TTrainDataForWord** train_data_for_words, int* vocabulary_size,
	TTrainDataForWord* train_data_for_silences);

int find_word(char* source_word, char** interesting_words, int number_of_interesting_words);

char* strip_line(char* source_line);

int load_interesting_words(char* filename, char*** interesting_words, int* number_of_interesting_words);

void finalize_interesting_words(char** interesting_words, int number_of_interesting_words);