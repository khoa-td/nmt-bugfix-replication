import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.MethodReferenceExpr;
import com.github.javaparser.ast.type.Type;
import java.util.*;


public class AstCollector {
    
    public final Set<String> types = new HashSet<>();
    public final Set<String> methods = new HashSet<>();

    public void parse(String code){
        CompilationUnit cu = JavaParser.parse("public class DummyClass {" + code + "}"); //wrap class for code and create AST tree

        for(Type t: cu.getNodesByType(Type.class)){ //find all Types in AST tree code
            types.addAll(Arrays.asList(filterString(t.asString()))); 
        }

        for(MethodDeclaration m: cu.getNodesByType(MethodDeclaration.class)){
            methods.add(m.getNameAsString());
        }

        for(MethodCallExpr mc: cu.getNodesByType(MethodCallExpr.class)){
            methods.add(mc.getNameAsString());
        }

        for (MethodCallExpr ms : cu.getNodesByType(MethodCallExpr.class)) {
            if (ms.getScope().isPresent()) {
                if (ms.getScope().get().isFieldAccessExpr()) {
                    String field = ms.getScope().get().toString();
                    String[] parts = field.split("\\.");
                    String[] letters = parts[parts.length - 1].split("");
                    if (!letters[0].equals(letters[0].toLowerCase()))
                        types.addAll(Arrays.asList(filterString(field)));
                }
                if (ms.getScope().get().isNameExpr()) {
                    String name = ms.getScope().get().toString();
                    String[] parts = name.split("\\.");
                    String[] letters = parts[parts.length - 1].split("");
                    if (!letters[0].equals(letters[0].toLowerCase()))
                        types.addAll(Arrays.asList(filterString(name)));
                }
            }
        }

        for(MethodReferenceExpr mr : cu.getNodesByType(MethodReferenceExpr.class)){
            methods.add(mr.getIdentifier());
        }
    }

    private String[] filterString(String s){
        s = s.replaceAll("\\[", " ").replaceAll("\\]", " ")
             .replaceAll("\\(", " ").replaceAll("\\)", " ")
             .replaceAll(">", " ").replaceAll("<", " ")
             .replaceAll("\\{", " ").replaceAll("\\}", " ")
             .replaceAll("\\,", " ").replaceAll("\\?", " ")
             .replaceAll("extends", " ").replaceAll("implements", " ");
        return s.split(" ");
    }

    public Set<String> getTypes()   { return types; }
    public Set<String> getMethods() { return methods; }
}
