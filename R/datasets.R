#' Self-Paced Reading Dataset on Chinese Relative Clauses
#'
#' This dataset contains data from a self-paced reading experiment on Chinese
#' relative clause comprehension. It is structured to support analysis of
#' reaction times, comprehension accuracy, and surprisal values across various
#' experimental conditions in a 2x2 fully crossed factorial design:
#'
#' - **Factor I**: Modification type (subject modification; object modification)
#' - **Factor II**: Relative clause type (subject relative; object relative)
#'
#' **Condition labels**:
#' - a) subject modification; subject relative
#' - b) subject modification; object relative
#' - c) object modification; subject relative
#' - d) object modification; object relative
#'
#' @format A tibble with 8,624 rows and 15 variables:
#' \describe{
#'   \item{subject}{Participant identifier, a character vector.}
#'   \item{item}{Trial item number, an integer.}
#'   \item{cond}{Experimental condition, a character vector indicating 
#'   variations in sentence structure (e.g., "a", "b", "c", "d").}
#'   \item{word}{Chinese word presented in each trial, a character vector.}
#'   \item{wordn}{Position of the word within the sentence, an integer.}
#'   \item{rt}{Reaction time in milliseconds for reading each word, 
#'   an integer.}
#'   \item{region}{Sentence region or phrase type (e.g., "hd1", "Det+CL"), 
#'   a character vector.}
#'   \item{question}{Comprehension question associated with the trial, a 
#'   character vector.}
#'   \item{accuracy}{Binary accuracy score for the comprehension question 
#'   (1 = correct, 0 = incorrect).}
#'   \item{correct_answer}{Expected correct answer for the comprehension 
#'   question, a character vector ("Y" or "N").}
#'   \item{question_type}{Type of comprehension question, a character vector.}
#'   \item{experiment}{Name of the experiment, indicating self-paced reading, a
#'   character vector.}
#'   \item{list}{Experimental list number, for counterbalancing item 
#'   presentation, an integer.}
#'   \item{sentence}{Full sentence used in the trial with words marked for 
#'   analysis, a character vector.}
#'   \item{surprisal}{Model-derived surprisal values for each word, a numeric 
#'   vector.}
#' }
#'
#' **Region codes in the dataset (column `region`)**:
#' - **N**: Main clause subject (in object-modifications only)
#' - **V**: Main clause verb (in object-modifications only)
#' - **Det+CL**: Determiner+classifier
#' - **Adv**: Adverb
#' - **VN**: RC-verb+RC-object (subject relatives) or RC-subject+RC-verb (object
#'  relatives)
#'     - Note: These two words were merged into one region after the experiment;
#'  they were presented as separate regions during the experiment.
#' - **FreqP**: Frequency phrase/durational phrase
#' - **DE**: Relativizer "de"
#' - **head**: Relative clause head noun
#' - **hd1**: First word after the head noun
#' - **hd2**: Second word after the head noun
#' - **hd3**: Third word after the head noun
#' - **hd4**: Fourth word after the head noun (only in subject-modifications)
#' - **hd5**: Fifth word after the head noun (only in subject-modifications)
#'
#' **Notes on reading times (column `rt`)**:
#' 
#' - The reading time of the relative clause region (e.g., "V-N" or "N-V") was 
#' computed by summing up the reading times of the relative clause verb and 
#' noun.
#' - The verb and noun were presented as two separate regions during the
#'  experiment.
#'
#' @source Jäger, L., Chen, Z., Li, Q., Lin, C.-J. C., & Vasishth, S. (2015).
#' \emph{The subject-relative advantage in Chinese: Evidence for 
#' expectation-based processing}.
#' Journal of Memory and Language, 79–80, 97-120. 
#' \url{https://doi.org/10.1016/j.jml.2014.10.005}
#' @family datasets
#' @usage data(df_jaeger14)
#' @examples
#' # Basic exploration
#' head(df_jaeger14)
#'
#' # Summarize reaction times by region
#'  library(tidytable)
#' df_jaeger14 |>
#'   group_by(region) |>
#'   summarize(mean_rt = mean(rt, na.rm = TRUE))
"df_jaeger14"


#' Example dataset: Two word-by-word tokenized sentences
#'
#' This dataset contains tokenized words from two example sentences, split 
#' word-by-word. It is structured to demonstrate the use of the `pangoling` 
#' package for processing text data.
#' package for processing text data.
#'
#' @format A data frame with 15 rows and 2 columns:
#' \describe{
#'   \item{sent_n}{(integer) Sentence number, indicating which sentence each 
#'                 word belongs to.}
#'   \item{word}{(character) Tokenized words from the sentences.}
#' }
#' @family datasets
#' @examples
#' # Load the dataset
#' data("df_sent")
#' df_sent

"df_sent"
