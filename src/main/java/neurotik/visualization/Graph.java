package neurotik.visualization;

import neurotik.tensor.Tensor;

import com.mxgraph.layout.hierarchical.mxHierarchicalLayout;
import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.util.mxConstants;
import com.mxgraph.view.mxGraph;

import javax.swing.*;
import java.util.HashMap;
import java.util.Map;

public class Graph {
    private mxGraph graph;
    private static Tensor lastNode;
    public Graph(Tensor lastNode){
        graph=new mxGraph();
        lastNode=lastNode;
        buildGraph(lastNode);
    }

    public void display(String title){

        graph.setCellsEditable(false);
        graph.setCellsMovable(true);
        mxGraphComponent graphComponent = new mxGraphComponent(graph);
        graph.setCellsEditable(false);
        graph.setCellsMovable(true);
        JFrame frame = new JFrame(title);
        frame.setResizable(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(graphComponent);
        frame.pack();
        frame.setVisible(true);
    }

    private void buildGraph(Tensor lastNode){
        Object parent = graph.getDefaultParent();
        Map<String, Object> vertexMap = new HashMap<>();
        try{
            graph.getModel().beginUpdate();
            setOptions();
            addNodes(lastNode, graph, parent, vertexMap);
            mxHierarchicalLayout layout = new mxHierarchicalLayout(graph);

            layout.setFineTuning(true);
            layout.setOrientation(SwingConstants.WEST);
            layout.execute(parent);

        }
        finally{
            graph.getModel().endUpdate();
        }
    }


    private void setOptions(){

        graph.setCellsEditable(true);
        graph.setCellsResizable(true);
        graph.setCellsMovable(true);
        graph.setCellsDeletable(false);
        graph.setEdgeLabelsMovable(false);
        graph.setHtmlLabels(true);
        graph.setAllowDanglingEdges(true);
        graph.setAutoSizeCells(true);
        graph.setCellsCloneable(false);
        graph.setCellsDisconnectable(false);
        graph.setCellsBendable(true);
        graph.setConnectableEdges(true);
        graph.setResetEdgesOnConnect(false);
        graph.setVertexLabelsMovable(false);
        graph.setSplitEnabled(false);
        graph.setKeepEdgesInForeground(true);
        graph.setMultigraph(false);
        setEdgeStyle();
        setVertexStyle();
        setInputVertexStyle();
        setValVertexStyle();
    }
    private void setEdgeStyle(){
        Map<String, Object> edgeStyle = graph.getStylesheet().getDefaultEdgeStyle();
        edgeStyle.put(mxConstants.STYLE_FONTSIZE,"12");
        edgeStyle.put(mxConstants.STYLE_LABEL_POSITION, mxConstants.ALIGN_TOP);
        edgeStyle.put(mxConstants.STYLE_LABEL_BORDERCOLOR, "#000000");
        edgeStyle.put(mxConstants.STYLE_EDGE, mxConstants.EDGESTYLE_ENTITY_RELATION);
        edgeStyle.put(mxConstants.STYLE_STROKECOLOR, "#000000");
        edgeStyle.put(mxConstants.STYLE_ROUNDED, true);
    }

    private void setVertexStyle(){
        Map<String, Object> vertexStyle =graph.getStylesheet().getDefaultVertexStyle();
        vertexStyle.put(mxConstants.STYLE_SHAPE, mxConstants.SHAPE_ELLIPSE);
        vertexStyle.put(mxConstants.STYLE_FILLCOLOR, "#FFFFFF");

        vertexStyle.put(mxConstants.STYLE_LABEL_POSITION, mxConstants.ALIGN_CENTER);
        vertexStyle.put(mxConstants.STYLE_VERTICAL_LABEL_POSITION, mxConstants.ALIGN_MIDDLE);
    }

    private void setInputVertexStyle(){
        Map<String, Object> inputsStyle=new HashMap<>();
        inputsStyle.put(mxConstants.STYLE_SHAPE, mxConstants.SHAPE_ELLIPSE);
        inputsStyle.put(mxConstants.STYLE_FILLCOLOR, "#FFFFFF");
        inputsStyle.put(mxConstants.STYLE_STROKECOLOR, "#FF0000");
        inputsStyle.put(mxConstants.STYLE_LABEL_POSITION, mxConstants.ALIGN_CENTER);
        inputsStyle.put(mxConstants.STYLE_VERTICAL_LABEL_POSITION, mxConstants.ALIGN_MIDDLE);
        graph.getStylesheet().putCellStyle("inputsStyle",inputsStyle);
    }

    private void setValVertexStyle(){
        Map<String, Object> vertexStyle=new HashMap<>();
        vertexStyle.put(mxConstants.STYLE_SHAPE, mxConstants.SHAPE_RECTANGLE);
        vertexStyle.put(mxConstants.STYLE_FILLCOLOR, "#FFFFFF");

        vertexStyle.put(mxConstants.STYLE_LABEL_POSITION, mxConstants.ALIGN_CENTER);
        vertexStyle.put(mxConstants.STYLE_VERTICAL_LABEL_POSITION, mxConstants.ALIGN_MIDDLE);
        graph.getStylesheet().putCellStyle("valStyle",vertexStyle);
    }



    private static void addNodes(Tensor node, mxGraph graph, Object parent, Map<String, Object> vertexMap) {
        Object vertex;
        Object vertex2;
        if (!vertexMap.containsKey(node.getId())) {
            if (node.getPrev().isEmpty()){
                vertex = graph.insertVertex(parent, node.getId(), node.getLabel(), 0, 0, 40, 40);
                graph.getModel().setStyle(vertex,"inputsStyle");
                vertex2 = graph.insertVertex(parent, node.getId()+1, "Label: \nGrad: ", 0, 0, 75, 40);
                graph.getModel().setStyle(vertex2,"valStyle");
            }
            else {
                vertex = graph.insertVertex(parent, node.getId(), node.getOperator(), 0, 0, 40, 40);
                graph.getModel().setStyle(vertex,"vertexStyle");
                vertex2 = graph.insertVertex(parent, node.getId()+1, "Label: \nGrad: ", 0, 0, 75, 40);
                graph.getModel().setStyle(vertex2,"valStyle");
            }
            graph.insertEdge(parent, null, "", vertex, vertex2, "edgeStyle");

            vertexMap.put(node.getId(), vertex);
            vertexMap.put(node.getId()+"additional", vertex2);

            for (Tensor prev : node.getPrev()) {
                addNodes(prev, graph, parent, vertexMap);
                Object edge;

                if (node==lastNode){
                    edge=graph.insertEdge(parent, null, null, vertexMap.get(prev.getId()+"additional"), vertex);
                }
                else{
                    edge=graph.insertEdge(parent, null,null, vertexMap.get(prev.getId()+"additional"), vertex);

                }

                graph.setCellStyle("edgeStyle",new Object[] { edge});
            }
        }
    }
}
