public class PreProcessor {
    public static String clean(String s) {
        s = s.replaceAll("(?:/\\*(?:[^*]|(?:\\*+[^*/]))*\\*+/)|(?://.*)", ""); 
        s = s.replaceAll("@.+", "");                                         
        return s;
    }
}
