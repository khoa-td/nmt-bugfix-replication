import edu.wm.cs.compiler.tools.generators.scanners.JavaLexer;
import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.Token;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;



public class Abstractor {
    private static final String SPACED_DOT = " . ";
    private final Set<String> idioms, types, methods;
    private final Map<String, String> typeMap = new HashMap<>(), methodMap = new HashMap<>(), varMap = new HashMap<>(), strMap = new HashMap<>(), charMap = new HashMap<>(),
        intMap = new HashMap<>(), floatMap = new HashMap<>();
    private int cType, cMethod, cVar, cStr, cChar, cInt, cFloat;

    public Abstractor(Set<String> idioms, Set<String> types, Set<String> methods) {
        this.idioms = idioms; this.types = types; this.methods = methods;
    }

    private List<Token> lex(String code) throws Exception {
        JavaLexer lexer = new JavaLexer(new ANTLRInputStream(new ByteArrayInputStream(code.getBytes(StandardCharsets.UTF_8))));
        List<Token> out = new ArrayList<>();
        for (Token t = lexer.nextToken(); t.getType() != Token.EOF; t = lexer.nextToken()) out.add(t);
        return out;
    }

    public String tokenize(String code) throws Exception{
        List<Token> tokens = lex(code);
        StringBuilder sb = new StringBuilder();

        for(int i = 0; i < tokens.size(); i++){
            String token;
            Token t = tokens.get(i);

            if(t.getType() == JavaLexer.Identifier){
                String name = t.getText();
                int j = i + 1;
                boolean need_dot = true;
                while(j < tokens.size()){
                    Token next = tokens.get(j);
                    if(need_dot){
                        if(next.getType() == JavaLexer.DOT){
                            name += next.getText();
                            need_dot = false;
                        }
                        else{
                            i = j - 1;
                            break;
                        }
                    }
                    else{
                        if(next.getType() == JavaLexer.Identifier){
                            name += next.getText();
                            need_dot = true;
                        }
                        else{
                            i = j - 1;
                            break;
                        }
                    }
                    j++;
                }
                token = analyzeIdentifier(name, tokens, i);

            }
            else if(t.getType() == JavaLexer.CharacterLiteral){
                token = literal(t.getText(), "CHAR_", charMap, 0);
            }
            else if(t.getType() == JavaLexer.FloatingPointLiteral) {
                token = literal(t.getText(), "FLOAT_", floatMap, 1);
            }
            else if (t.getType() == JavaLexer.IntegerLiteral) {
                token = literal(t.getText(), "INT_", intMap, 2);
            }
            else if (t.getType() == JavaLexer.StringLiteral) {
                token = literal(t.getText(), "STRING_", strMap, 3);
            }
            else {
                token = t.getText();
            }
            sb.append(token + " ");
        }
        return sb.toString().trim();
    }

    private String analyzeIdentifier(String token, List<Token> tokens, int i) {
        if (idioms.contains(token)) return token;

        String[] parts = token.split("\\.");
        if (parts.length > 1) {
            String last = parts[parts.length - 1];  // head code
            String first = token.substring(0, token.length() - last.length() - 1);  // tail code
            if (idioms.contains(last)) {
                if (idioms.contains(first)) return first + SPACED_DOT + last;
                else if (types.contains(first)) return typeId(first) + SPACED_DOT + last;
                else return varId(first) + SPACED_DOT + last;
            }
        }

        if (types.contains(token)) return typeId(token);

        // process funtion(Ex: cal(), Try(), ...)
        boolean couldBeMethod = false;
        if (i + 1 < tokens.size() && tokens.get(i + 1).getType() == JavaLexer.LPAREN) couldBeMethod = true;
        if (i > 2 && tokens.get(i - 1).getType() == JavaLexer.COLON && tokens.get(i - 2).getType() == JavaLexer.COLON) couldBeMethod = true;

        if (methods.contains(token) && couldBeMethod) return methodId(token);

        // process function call by a object(Ex: A.cal(), Math.max(), ...)
        if (parts.length > 1) {
            String last = parts[parts.length - 1];
            String first = token.substring(0, token.length() - last.length() - 1);
            if (methods.contains(last) && couldBeMethod) {
                if (idioms.contains(first)) return first + SPACED_DOT + methodId(last);
                else if (types.contains(first)) return typeId(first) + SPACED_DOT + methodId(last);
                else return varId(first) + SPACED_DOT + methodId(last);
            }
        }
        return varId(token);
    }

    private String typeId(String s)   { return typeMap.computeIfAbsent(s, k -> "TYPE_" + (++cType)); }
    private String methodId(String s) { return methodMap.computeIfAbsent(s, k -> "METHOD_" + (++cMethod)); }
    private String varId(String s)    { return varMap.computeIfAbsent(s, k -> "VAR_" + (++cVar)); }

    private String literal(String text, String prefix, Map<String, String> map, int kind){
        if(idioms.contains(text)) return text;

        if (map.containsKey(text)) return map.get(text);
        int term = 0;
        if(kind == 0){
            cChar++;
            term = cChar;
        }
        else if(kind == 1){
            cFloat++;
            term = cFloat;
        }
        else if(kind == 2){
            cInt++;
            term = cInt;
        }
        else{
            cStr++;
            term = cStr;
        }

        String id = prefix + term;
        map.put(text, id);
        return id;
    }
}
